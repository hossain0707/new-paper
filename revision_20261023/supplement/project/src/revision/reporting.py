"""Matched-prediction summaries; patients are the independent inferential units."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
from src.evaluation.metrics import summarize
from src.revision.experiment import BASE, SEEDS, simplex_weights

HORIZONS=(30,60,90)
STACK=('RandomForest','XGBoost','TFTEncoder','BiLSTMTransformer','CNNGRU')


def paired_effect(a,b,seed=917,nboot=10000):
    """Mean paired subject-MAE difference and percentile patient bootstrap CI."""
    diff=np.asarray(a,float)-np.asarray(b,float)
    rng=np.random.default_rng(seed)
    boot=diff[rng.integers(0,len(diff),size=(nboot,len(diff)))].mean(axis=1)
    p=1. if np.allclose(diff,0) else float(wilcoxon(diff,zero_method='pratt',method='auto').pvalue)
    return {'n_patients':len(diff),'difference_MAE':float(diff.mean()),
            'ci95_low':float(np.quantile(boot,.025)),'ci95_high':float(np.quantile(boot,.975)),
            'p_raw':p,'patients_improved':int((diff<0).sum())}


def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out


def collect(data_dir,out_dir,datasets=('ddatshr','diadata')):
    data_dir=Path(data_dir);out_dir=Path(out_dir);predictions=[];stack_rows=[];timings=[]
    for dataset in datasets:
        meta=pd.read_parquet(data_dir/f'{dataset}_samples.parquet');meta.PtID=meta.PtID.astype(str)
        data=np.load(data_dir/f'{dataset}.npz');x=data['X'];truth=data['y']
        for condition in ('cgm_hr','cgm'):
            for patient in sorted(meta.PtID.unique()):
                seed_frames={}
                for seed in SEEDS:
                    frames={}
                    for model in BASE:
                        path=out_dir/'fits'/dataset/condition/model/f'seed_{seed}'/patient
                        if not (path/'complete.json').exists():raise FileNotFoundError(f'Incomplete fit: {path}')
                        manifest=json.loads((path/'complete.json').read_text());timings.append(manifest)
                        frames[model]={s:pd.read_parquet(path/f'{s}_predictions.parquet').sort_values('sample_id').reset_index(drop=True)
                                       for s in ('test','validation')}
                    for model,members in [('Stack',STACK),('TreeStack',('RandomForest','XGBoost'))]:
                        reference=frames[members[0]]
                        frames[model]={s:reference[s].copy() for s in reference}
                        for h in HORIZONS:
                            for s in reference:
                                for member in members:
                                    assert reference[s].sample_id.equals(frames[member][s].sample_id)
                                    np.testing.assert_array_equal(reference[s][f'true_{h}'],frames[member][s][f'true_{h}'])
                            pv=np.column_stack([frames[m]['validation'][f'pred_{h}'] for m in members])
                            yv=reference['validation'][f'true_{h}'].to_numpy()
                            # A common scale improves numerical conditioning without changing the optimum.
                            w=simplex_weights(pv/100,yv/100)
                            for s in reference:
                                frames[model][s][f'pred_{h}']=np.column_stack([frames[m][s][f'pred_{h}'] for m in members])@w
                            for member,weight in zip(members,w):
                                stack_rows.append(dict(dataset=dataset,condition=condition,PtID=patient,seed=seed,
                                                       model=model,horizon=h,member=member,weight=float(weight)))
                    seed_frames[seed]=frames
                    for model in frames:
                        f=frames[model]['test'].copy();f['seed']=seed;f['model']=model
                        f['condition']=condition;f['dataset']=dataset;predictions.append(f)
                for model in seed_frames[SEEDS[0]]:
                    f=seed_frames[SEEDS[0]][model]['test'].copy()
                    for h in HORIZONS:
                        f[f'pred_{h}']=np.mean([seed_frames[s][model]['test'][f'pred_{h}'] for s in SEEDS],axis=0)
                        v=seed_frames[SEEDS[0]][model]['validation']
                        vp=np.mean([seed_frames[s][model]['validation'][f'pred_{h}'] for s in SEEDS],axis=0)
                        radius=float(np.quantile(np.abs(v[f'true_{h}']-vp),.95,method='higher'))
                        f[f'lower_{h}']=f[f'pred_{h}']-radius;f[f'upper_{h}']=f[f'pred_{h}']+radius
                    f['seed']='mean';f['model']=model;f['condition']=condition;f['dataset']=dataset;predictions.append(f)
            for model in ('Persistence','LinearTrend'):
                f=meta.copy()
                for j,h in enumerate(HORIZONS):
                    f[f'true_{h}']=truth[:,j]
                    # Trend extrapolation is intentionally unclipped and uses the last two available bins.
                    slope=(x[:,-1,0]-x[:,-2,0])/15
                    lead=h+meta.get('input_lag_minutes',pd.Series(0,index=meta.index)).to_numpy()
                    f[f'pred_{h}']=x[:,-1,0]+(lead*slope if model=='LinearTrend' else 0)
                f['seed']='mean';f['model']=model;f['condition']=condition;f['dataset']=dataset;predictions.append(f)
    frame=pd.concat(predictions,ignore_index=True);frame.seed=frame.seed.astype(str)
    tables=out_dir/'tables';tables.mkdir(exist_ok=True)
    frame.to_parquet(out_dir/'all_predictions.parquet',index=False)
    pd.DataFrame(stack_rows).to_csv(tables/'stack_weights.csv',index=False)
    pd.json_normalize(timings).to_csv(tables/'fit_manifest.csv',index=False)
    return frame


def summarize_predictions(frame,out_dir):
    out_dir=Path(out_dir);tables=out_dir/'tables';tables.mkdir(exist_ok=True)
    per=[];ranges=[];episodes=[];pooled=[]
    for keys,f in frame.groupby(['dataset','condition','model','seed','PtID']):
        info=dict(zip(['dataset','condition','model','seed','PtID'],keys))
        for h in HORIZONS:
            y=f[f'true_{h}'].to_numpy();p=f[f'pred_{h}'].to_numpy()
            row={**info,'horizon':h,**summarize(y,p)}
            if f.get(f'lower_{h}',pd.Series(dtype=float)).notna().any():
                row['interval_coverage']=float(((y>=f[f'lower_{h}'])&(y<=f[f'upper_{h}'])).mean())
                row['interval_width']=float((f[f'upper_{h}']-f[f'lower_{h}']).mean())
            per.append(row)
            if keys[3]!='mean':continue
            for state,mask in [('low',y<70),('in_range',(y>=70)&(y<=180)),('high',y>180)]:
                if mask.any():ranges.append({**info,'horizon':h,'range':state,**summarize(y[mask],p[mask])})
            for event,actual,forecast in [('low',y<70,p<70),('high',y>180,p>180)]:
                tp=int((actual&forecast).sum());fp=int((~actual&forecast).sum());fn=int((actual&~forecast).sum())
                episodes.append({**info,'horizon':h,'event':event,'tp':tp,'fp':fp,'fn':fn,
                                 'sensitivity':tp/(tp+fn) if tp+fn else np.nan,
                                 'precision':tp/(tp+fp) if tp+fp else np.nan})
    per=pd.DataFrame(per);per.to_csv(tables/'per_patient_metrics.csv',index=False)
    pd.DataFrame(ranges).to_csv(tables/'glycemic_range_metrics.csv',index=False)
    pd.DataFrame(episodes).to_csv(tables/'target_classification.csv',index=False)
    macro=per.groupby(['dataset','condition','model','seed','horizon']).agg(
        n_patients=('PtID','nunique'),n_samples=('n','sum'),MAE=('MAE','mean'),RMSE=('RMSE','mean'),
        Clarke_A_pct=('Clarke_A_pct','mean'),Clarke_B_pct=('Clarke_B_pct','mean'),
        Clarke_C_pct=('Clarke_C_pct','mean'),Clarke_D_pct=('Clarke_D_pct','mean'),Clarke_E_pct=('Clarke_E_pct','mean'),
        interval_coverage=('interval_coverage','mean'),interval_width=('interval_width','mean')).reset_index()
    macro.to_csv(tables/'subject_macro_metrics.csv',index=False)
    for keys,f in frame[frame.seed.eq('mean')].groupby(['dataset','condition','model']):
        for h in HORIZONS:
            pooled.append(dict(zip(['dataset','condition','model'],keys))|{'horizon':h}|summarize(f[f'true_{h}'],f[f'pred_{h}']))
    pd.DataFrame(pooled).to_csv(tables/'pooled_metrics.csv',index=False)
    comparisons=[]
    pm=per[per.seed.eq('mean')]
    for dataset in pm.dataset.unique():
        for h in HORIZONS:
            d=pm[(pm.dataset==dataset)&(pm.horizon==h)]
            for label,a,ac,b,bc in [('HR_added','RandomForest','cgm_hr','RandomForest','cgm'),
                                   ('Stack_vs_RF','Stack','cgm_hr','RandomForest','cgm_hr'),
                                   ('RF_vs_persistence','RandomForest','cgm_hr','Persistence','cgm_hr')]:
                aa=d[(d.model==a)&(d.condition==ac)].set_index('PtID').MAE.sort_index()
                bb=d[(d.model==b)&(d.condition==bc)].set_index('PtID').MAE.sort_index()
                assert aa.index.equals(bb.index)
                comparisons.append({'dataset':dataset,'horizon':h,'comparison':label,**paired_effect(aa,bb)})
    c=pd.DataFrame(comparisons)
    # One family includes all prespecified tests across both datasets and all horizons.
    c['p_holm']=holm(c.p_raw);c.to_csv(tables/'primary_comparisons.csv',index=False)
    return macro,c
