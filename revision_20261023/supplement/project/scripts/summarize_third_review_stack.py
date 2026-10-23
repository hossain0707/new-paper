"""Require complete matched Stack predictions; report new sensitivity separately."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    config=json.loads((args.input/'protocol.json').read_text());assert not config['pilot']
    rows=[];manifest=[]
    for cohort in ['ddatshr','diadata']:
        meta=pd.read_parquet(ROOT/f'data/processed/revision_v2/history_120/{cohort}_samples.parquet')
        for pt in sorted(meta.PtID.astype(str).unique()):
            predictions={}
            for model in ['RandomForest','Stack']:
                frames=[]
                for seed in [42,1729,2026]:
                    folder=args.input/(f'validation/stack/{cohort}/seed_{seed}/{pt}' if model=='Stack' else f'validation/fits/{cohort}/cgm_hr/RandomForest/seed_{seed}/{pt}')
                    path=folder/'test.parquet';frame=pd.read_parquet(path).sort_values('sample_id').reset_index(drop=True)
                    manifest.append({'file':str(path.relative_to(args.input)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                    if frames:assert frame.sample_id.equals(frames[0].sample_id)
                    assert set(frame.sample_id)==set(meta[meta.PtID.astype(str).eq(pt)].sample_id)
                    frames.append(frame)
                mean=frames[0].copy()
                for h in [30,60,90]:mean[f'pred_{h}']=np.mean([f[f'pred_{h}'].to_numpy() for f in frames],axis=0)
                predictions[model]=mean
                for h in [30,60,90]:
                    rows.append({'dataset':cohort,'PtID':pt,'model':model,'horizon':h,'n':len(mean),
                                 'MAE':float(np.abs(mean[f'true_{h}']-mean[f'pred_{h}']).mean()),
                                 'RMSE':float(np.sqrt(np.mean((mean[f'true_{h}']-mean[f'pred_{h}'])**2)))})
            assert predictions['RandomForest'].sample_id.equals(predictions['Stack'].sample_id)
    args.output.mkdir(parents=True,exist_ok=False)
    per=pd.DataFrame(rows);per.to_csv(args.output/'per_patient_metrics.csv',index=False)
    macro=per.groupby(['dataset','model','horizon']).agg(MAE=('MAE','mean'),n_people=('PtID','nunique')).reset_index()
    macro.to_csv(args.output/'subject_macro_metrics.csv',index=False)
    historical=pd.read_csv(ROOT/'results/revision_v2/tables/per_patient_metrics.csv',dtype={'seed':str,'PtID':str})
    historical=historical[historical.seed.eq('mean')&historical.condition.eq('cgm_hr')]
    contrasts=[]
    for (cohort,h),group in per.groupby(['dataset','horizon']):
        pivot=group.pivot(index='PtID',columns='model',values='MAE')
        for label,a,b in [('Stack_three_minus_RF_three',pivot.Stack,pivot.RandomForest),
                          ('Stack_three_minus_Stack_one',pivot.Stack,historical[historical.dataset.eq(cohort)&historical.horizon.eq(h)&historical.model.eq('Stack')].set_index('PtID').MAE.reindex(pivot.index)),
                          ('RF_three_minus_RF_one',pivot.RandomForest,historical[historical.dataset.eq(cohort)&historical.horizon.eq(h)&historical.model.eq('RandomForest')].set_index('PtID').MAE.reindex(pivot.index))]:
            assert a.notna().all() and b.notna().all();d=(a-b).to_numpy();rng=np.random.default_rng(917)
            boot=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(axis=1)
            contrasts.append({'dataset':cohort,'horizon':h,'contrast':label,'n_people':len(d),
                              'difference_MAE':d.mean(),'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975)})
    pd.DataFrame(contrasts).to_csv(args.output/'descriptive_comparisons.csv',index=False)
    count=len(list((args.input/'validation/fits').glob('*/*/*/*/*/complete.json')))
    assert count==690,(count,'complete base fits required')
    (args.output/'provenance.json').write_text(json.dumps({'complete_base_fits':count,'prediction_manifest':manifest,
      'analysis':'post-review descriptive sensitivity; new Spark environment; no primary-family replacement',
      'caution':'Validation coverage and base-training size both change; shared folds limit independence.',
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    print(macro.to_string(index=False));print(pd.DataFrame(contrasts).to_string(index=False))

if __name__=='__main__':main()
