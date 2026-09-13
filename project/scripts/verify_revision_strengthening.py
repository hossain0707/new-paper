"""Independent checks of new split coverage, targets, scalers, metrics and freezing."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,joblib
from src.revision.data import sha256
p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');p.add_argument('--final',action='store_true');args=p.parse_args()
out=ROOT/'results/revision_v3';checks=[]
freeze=json.loads((out/'external/freeze.json').read_text())
for name,digest in freeze['files'].items():assert sha256(ROOT/name)==digest,name
started=json.loads((out/'external/scoring_started.json').read_text());assert freeze['created_utc']<started['utc'];assert started['freeze_sha256']==sha256(out/'external/freeze.json')
checks.append('Development artifacts and stack weights frozen before AZT1D scoring; hashes unchanged')
count=0
for dataset in ['ddatshr','diadata']:
 data=ROOT/'data/processed/revision_v2/history_120';arr=np.load(data/f'{dataset}.npz');meta=pd.read_parquet(data/f'{dataset}_samples.parquet');meta.PtID=meta.PtID.astype(str);indexed=meta.assign(idx=np.arange(len(meta))).set_index('sample_id')
 for condition in ['cgm','cgm_hr']:
  for seed in [42,1729,2026]:
   ids=[]
   for path in sorted((out/'validation/fits'/dataset/condition/'RandomForest'/f'seed_{seed}').glob('*/complete.json')):
    m=json.loads(path.read_text());i=m['identity'];count+=1
    train,val,test=map(set,[i['train'],i['validation'],i['test']]);assert len(val)==3 and len(test)==1
    assert not train&val and not train&test and not val&test;assert train|val|test==set(meta.PtID)
    assert i['data_sha256']==sha256(data/f'{dataset}.npz') and i['code_sha256']==sha256(ROOT/'src/revision/strengthening.py')
    f=pd.read_parquet(path.parent/'test.parquet');ids.extend(f.sample_id);ix=indexed.loc[f.sample_id,'idx'].to_numpy();assert set(f.PtID.astype(str))==test
    np.testing.assert_array_equal(f[[f'true_{h}' for h in [30,60,90]]],arr['y'][ix])
    v=pd.read_parquet(path.parent/'validation.parquet');assert set(v.PtID.astype(str))==val
   assert len(ids)==len(set(ids))==len(meta) and set(ids)==set(meta.sample_id)
assert count==276;checks.append('All 276 sensitivity fits: three validation patients, disjoint splits, complete unchanged test origins and observed targets')
arr=np.load(ROOT/'data/processed/revision_v3/azt1d.npz');X=arr['X'];Y=arr['y'];meta=pd.read_parquet(ROOT/'data/processed/revision_v3/azt1d_samples.parquet');indexed=meta.assign(idx=np.arange(len(meta))).set_index('sample_id')
f=pd.read_parquet(out/'external/all_predictions.parquet');assert meta.PtID.nunique()==24 and 'AZT14' not in set(meta.PtID)
metrics=pd.read_csv(out/'tables/per_patient_metrics.csv',dtype={'seed':str,'PtID':str}).set_index(['dataset','condition','model','seed','PtID','horizon'])
for keys,g in f.groupby(['dataset','condition','model','seed']):
 assert len(g)==len(meta) and g.sample_id.is_unique and set(g.sample_id)==set(meta.sample_id)
 ix=indexed.loc[g.sample_id,'idx'].to_numpy();np.testing.assert_array_equal(g[[f'true_{h}' for h in [30,60,90]]],Y[ix])
 for pt,d in g.groupby('PtID'):
  for h in [30,60,90]:
   e=d[f'pred_{h}'].to_numpy(dtype=float)-d[f'true_{h}'].to_numpy(dtype=float);r=metrics.loc[(*keys,str(pt),h)]
   assert abs(np.abs(e).mean()-r.MAE)<1e-4 and abs(np.sqrt((e*e).mean())-r.RMSE)<1e-4
for seed in [42,1729,2026]:
 for h in [30,60,90]:
  w=np.array(freeze['weights'][str(seed)][str(h)]);assert (w>=0).all();np.testing.assert_allclose(w.sum(),1,atol=1e-7)
  pred=np.column_stack([f[(f.model==m)&(f.seed==str(seed))][f'pred_{h}'] for m in freeze['members']])@w
  np.testing.assert_allclose(pred,f[(f.model=='Stack')&(f.seed==str(seed))][f'pred_{h}'],atol=1e-5)
checks.append('Every AZT1D model/seed has identical complete origins and targets; all patient MAE/RMSE and stack predictions independently recalculated')
if args.gpu:
 import torch
 from src.revision.experiment import BASE,make_deep,seed_all
 for seed in [42,1729,2026]:
  for name in BASE:
   path=out/'external/fits/ddatshr/glucose_only'/name/f'seed_{seed}/development';xx=X[:2048]
   if (path/'checkpoint.joblib').exists():
    saved=joblib.load(path/'checkpoint.joblib');p=saved['model'].predict(saved['scaler'].transform(xx.reshape(len(xx),-1)))
   else:
    seed_all(seed);torch.set_num_threads(1);saved=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False);net,isq=make_deep(name,1);net.load_state_dict(saved['state_dict']);net.to('cuda:1').eval()
    with torch.no_grad():
     p=net(torch.tensor((xx-saved['feature_mean'])/saved['feature_std'],device='cuda:1'));p=p[:,:,1] if isq else p
    p=p.cpu().numpy()*saved['target_std']+saved['target_mean'];del net;torch.cuda.empty_cache()
   expected=f[(f.model==name)&(f.seed==str(seed))].iloc[:len(xx)]
   np.testing.assert_allclose(p+xx[:,-1,0,None],expected[[f'pred_{h}' for h in [30,60,90]]],atol=1e-4,rtol=1e-6)
 checks.append('All 18 external checkpoints reloaded and reproduced original 2048-example inference batch')
 (out/'checkpoint_verification.json').write_text(json.dumps({'status':'passed','checks':checks,'freeze_sha256':sha256(out/'external/freeze.json'),'predictions_sha256':sha256(out/'external/all_predictions.parquet')},indent=2))
if args.final:
 phase=json.loads((out/'checkpoint_verification.json').read_text());assert phase['status']=='passed'
 assert phase['freeze_sha256']==sha256(out/'external/freeze.json') and phase['predictions_sha256']==sha256(out/'external/all_predictions.parquet')
 chrono=json.loads((out/'chronos/verification.json').read_text());assert chrono['status']=='passed'
 assert chrono['freeze_sha256']==sha256(out/'chronos/freeze.json')
 for dataset,digest in chrono['predictions_sha256'].items():assert digest==sha256(out/f'chronos/{dataset}_predictions.parquet')
 assert json.loads((out/'external/raw_boundary_verification.json').read_text())['status']=='passed'
 from src.revision.grounding import validate
 responses=list((out/'latency/responses').glob('*.json'));assert len(responses)==93
 for p in responses:
  r=json.loads(p.read_text());case=json.loads((ROOT/'results/revision_v2/llm/cases'/f"{r['case_id']}.json").read_text());assert validate(r['raw_text'],case)['accepted']==r['validation']['accepted']
  assert abs(sum(r[k] for k in ['prediction_s','shap_s','generation_s','validation_render_s'])-r['total_s'])<1e-8
 checks.append('All 90 latency observations plus 3 labelled warmups retained; timing sums and gate decisions independently verified')
 (out/'artifact_verification.json').write_text(json.dumps({'status':'passed','checks':checks,'tables':{p.name:sha256(p) for p in (out/'tables').glob('*.csv')},'primary_experiment_sha256':sha256(ROOT/'src/revision/experiment.py')},indent=2))
print('\n'.join(checks))
