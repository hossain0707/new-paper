"""Descriptive additional evidence; never alters the primary multiplicity family."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.evaluation.metrics import summarize
from src.revision.data import sha256
out=ROOT/'results/revision_v3';tables=out/'tables';tables.mkdir(exist_ok=True)
frames=[];manifests=[]
for dataset in ['ddatshr','diadata']:
 for condition in ['cgm','cgm_hr']:
  folder=out/'validation/fits'/dataset/condition/'RandomForest'
  seeds=[]
  for p in sorted(folder.glob('seed_*/*/complete.json')):
   m=json.loads(p.read_text());manifests.append(m)
   f=pd.read_parquet(p.parent/'test.parquet');f['seed']=p.parent.parent.name.removeprefix('seed_');seeds.append(f)
  f=pd.concat(seeds,ignore_index=True)
  assert f.groupby('sample_id').size().eq(3).all()
  mean=f.groupby(['sample_id','PtID'],as_index=False)[[f'{k}_{h}' for k in ['true','pred'] for h in [30,60,90]]].mean();mean['seed']='mean'
  f=pd.concat([f,mean],ignore_index=True);f['dataset']=dataset;f['condition']=condition;f['model']='RF_ThreeValidation';frames.append(f)
assert len(manifests)==276
pd.json_normalize(manifests).to_csv(tables/'validation_fit_manifest.csv',index=False)
for dataset in ['ddatshr','diadata']:frames.append(pd.read_parquet(out/f'chronos/{dataset}_predictions.parquet'))
frames.append(pd.read_parquet(out/'external/all_predictions.parquet'))
per=[];ranges=[];pooled=[]
for f in frames:
 for keys,g in f.groupby(['dataset','condition','model','seed','PtID']):
  info=dict(zip(['dataset','condition','model','seed','PtID'],keys))
  for h in [30,60,90]:
   y=g[f'true_{h}'].to_numpy();p=g[f'pred_{h}'].to_numpy();per.append(info|{'horizon':h}|summarize(y,p))
   if str(info['seed'])=='mean':
    for label,mask in [('low',y<70),('in_range',(y>=70)&(y<=180)),('high',y>180)]:
     if mask.any():ranges.append(info|{'horizon':h,'range':label}|summarize(y[mask],p[mask]))
 for keys,g in f[f.seed.astype(str).eq('mean')].groupby(['dataset','condition','model']):
  for h in [30,60,90]:pooled.append(dict(zip(['dataset','condition','model'],keys))|{'horizon':h}|summarize(g[f'true_{h}'],g[f'pred_{h}']))
per=pd.DataFrame(per);per.to_csv(tables/'per_patient_metrics.csv',index=False)
pd.DataFrame(ranges).to_csv(tables/'glycemic_range_metrics.csv',index=False);pd.DataFrame(pooled).to_csv(tables/'pooled_metrics.csv',index=False)
macro=[]
for keys,g in per.groupby(['dataset','condition','model','seed','horizon']):
 info=dict(zip(['dataset','condition','model','seed','horizon'],keys));r=info|{'n_patients':len(g),'n_samples':int(g.n.sum())}
 for metric in ['MAE','RMSE','Clarke_A_pct','Clarke_B_pct','Clarke_C_pct','Clarke_D_pct','Clarke_E_pct']:
  a=g[metric].to_numpy();r[metric]=a.mean()
  if metric in ['MAE','RMSE']:
   ix=np.random.default_rng(917).integers(0,len(a),(10000,len(a)));ci=np.quantile(a[ix].mean(axis=1),[.025,.975]);r[metric+'_ci_low'],r[metric+'_ci_high']=ci
 macro.append(r)
pd.DataFrame(macro).to_csv(tables/'subject_macro_metrics.csv',index=False)
old=pd.read_csv(ROOT/'results/revision_v2/tables/per_patient_metrics.csv',dtype={'PtID':str,'seed':str})
per.PtID=per.PtID.astype(str);per.seed=per.seed.astype(str);pairs=[]
def compare(a,b,label,dataset,h):
 a=a.set_index('PtID').MAE.sort_index();b=b.set_index('PtID').MAE.sort_index();assert a.index.equals(b.index)
 d=(a-b).to_numpy();idx=np.random.default_rng(917).integers(0,len(d),(10000,len(d)));ci=np.quantile(d[idx].mean(axis=1),[.025,.975])
 pairs.append(dict(dataset=dataset,horizon=h,comparison=label,n_patients=len(d),difference_MAE=d.mean(),ci95_low=ci[0],ci95_high=ci[1],patients_improved=int((d<0).sum())))
for dataset in ['ddatshr','diadata']:
 for h in [30,60,90]:
  a=per[(per.dataset==dataset)&(per.horizon==h)&(per.seed=='mean')];b=old[(old.dataset==dataset)&(old.horizon==h)&(old.seed=='mean')]
  for c in ['cgm','cgm_hr']:compare(a[(a.model=='RF_ThreeValidation')&(a.condition==c)],b[(b.model=='RandomForest')&(b.condition==c)],'RF_three_vs_one_validation_'+c,dataset,h)
  compare(a[(a.model=='RF_ThreeValidation')&(a.condition=='cgm_hr')],a[(a.model=='RF_ThreeValidation')&(a.condition=='cgm')],'HR_added_three_validation',dataset,h)
  for model in ['RandomForest','Ridge','TFTEncoder','Stack','Persistence']:compare(a[a.model=='Chronos2'],b[(b.model==model)&(b.condition=='cgm')],'Chronos2_vs_'+model+'_cgm',dataset,h)
pd.DataFrame(pairs).to_csv(tables/'descriptive_comparisons.csv',index=False)
(out/'summary_complete.json').write_text(json.dumps({'primary_results_unchanged':True,'additional_p_values':False,'bootstrap':'10000 participant resamples; descriptive; shared training folds not independent','table_hashes':{p.name:sha256(p) for p in tables.glob('*.csv')}},indent=2))
print(pd.DataFrame(macro).query("seed == 'mean'")[['dataset','condition','model','horizon','MAE']].to_string(index=False))
