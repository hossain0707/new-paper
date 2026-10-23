"""Recalculate RF hypoglycemia and interval denominators from recovered predictions."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
fits=ROOT/'results/revision_v4/recovered_rf_primary'
assert json.loads((fits/'VERIFIED.json').read_text())['fits']==138
out=ROOT/'results/revision_v4/rf_raw_verification';out.mkdir(parents=True,exist_ok=False)
old=pd.read_csv(ROOT/'results/revision_v2/tables/per_patient_metrics.csv',dtype={'seed':str,'PtID':str})
old=old[old.seed.eq('mean')&old.model.eq('RandomForest')&old.condition.eq('cgm_hr')]
classes=pd.read_csv(ROOT/'results/revision_v2/tables/target_classification.csv',dtype={'seed':str,'PtID':str})
classes=classes[classes.seed.eq('mean')&classes.model.eq('RandomForest')&classes.condition.eq('cgm_hr')&classes.event.eq('low')]
people=[];prediction_frames=[]
for cohort in ['ddatshr','diadata']:
    meta=pd.read_parquet(ROOT/f'data/processed/revision_v2/history_120/{cohort}_samples.parquet')
    for pt in sorted(meta.PtID.astype(str).unique()):
        frames={}
        for split in ['test','validation']:
            frames[split]=[pd.read_parquet(fits/f'fits/{cohort}/cgm_hr/RandomForest/seed_{s}/{pt}/{split}_predictions.parquet').sort_values('sample_id').reset_index(drop=True) for s in [42,1729,2026]]
            assert all(f.sample_id.equals(frames[split][0].sample_id) for f in frames[split])
        f=frames['test'][0].copy();f['dataset']=cohort
        for h in [30,60,90]:
            y=f[f'true_{h}'].to_numpy();pred=np.mean([d[f'pred_{h}'].to_numpy() for d in frames['test']],axis=0)
            vp=np.mean([d[f'pred_{h}'].to_numpy() for d in frames['validation']],axis=0)
            radius=float(np.quantile(np.abs(frames['validation'][0][f'true_{h}'].to_numpy()-vp),.95,method='higher'))
            f[f'pred_{h}']=pred;f[f'lower_{h}']=pred-radius;f[f'upper_{h}']=pred+radius
            covered=int(((y>=pred-radius)&(y<=pred+radius)).sum());actual=y<70;positive=pred<70
            tp=int((actual&positive).sum());fp=int((~actual&positive).sum());fn=int((actual&~positive).sum())
            saved=old[old.dataset.eq(cohort)&old.PtID.eq(pt)&old.horizon.eq(h)].iloc[0]
            event=classes[classes.dataset.eq(cohort)&classes.PtID.eq(pt)&classes.horizon.eq(h)].iloc[0]
            assert [tp,fp,fn]==[event.tp,event.fp,event.fn]
            assert abs(covered/len(f)-saved.interval_coverage)<1e-12
            assert abs(2*radius-saved.interval_width)<1e-7
            assert abs(float(np.abs(y-pred).mean())-saved.MAE)<1e-7
            people.append({'dataset':cohort,'PtID':pt,'horizon':h,'n':len(f),'covered':covered,'interval_radius':radius,'coverage':covered/len(f),'tp':tp,'fp':fp,'fn':fn})
        prediction_frames.append(f)
pd.concat(prediction_frames,ignore_index=True).to_parquet(out/'rf_seed_mean_predictions.parquet',index=False)
people=pd.DataFrame(people);people.to_csv(out/'participant_counts.csv',index=False)
summary=people.groupby(['dataset','horizon']).agg(n_people=('PtID','nunique'),n_origins=('n','sum'),covered=('covered','sum'),coverage_macro=('coverage','mean'),tp=('tp','sum'),fp=('fp','sum'),fn=('fn','sum')).reset_index()
summary['coverage_pooled']=summary.covered/summary.n_origins
summary['sensitivity_pooled']=summary.tp/(summary.tp+summary.fn)
summary['precision_pooled']=summary.tp/(summary.tp+summary.fp)
summary.to_csv(out/'cohort_summary.csv',index=False)
(out/'VERIFIED.json').write_text(json.dumps({'participant_horizon_checks':len(people),'status':'all saved hypoglycemia counts, interval fractions/widths and RF mean-seed MAEs match recovered raw predictions','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
print(summary.to_string(index=False))
