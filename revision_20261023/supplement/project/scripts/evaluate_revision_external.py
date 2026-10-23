"""Score AZT1D with previously frozen DDATSHR models; no external adaptation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib,torch
from src.revision.experiment import BASE,SEEDS,make_deep,seed_all
from src.revision.data import sha256
out=ROOT/'results/revision_v3/external';freeze=json.loads((out/'freeze.json').read_text())
for name,digest in freeze['files'].items():assert sha256(ROOT/name)==digest,name
assert not (out/'all_predictions.parquet').exists(),'Preserve completed scoring'
(out/'scoring_started.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':sha256(out/'freeze.json'),'script_sha256':sha256(Path(__file__))},indent=2))
data=ROOT/'data/processed/revision_v3';arr=np.load(data/'azt1d.npz');X=arr['X'];Y=arr['y'];meta=pd.read_parquet(data/'azt1d_samples.parquet');preds={};rows=[]
for seed in SEEDS:
 for name in BASE:
  path=out/'fits/ddatshr/glucose_only'/name/f'seed_{seed}/development'
  if (path/'checkpoint.joblib').exists():
   saved=joblib.load(path/'checkpoint.joblib');delta=saved['model'].predict(saved['scaler'].transform(X.reshape(len(X),-1)))
  else:
   seed_all(seed);torch.set_num_threads(1);saved=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
   model,isq=make_deep(name,1);model.load_state_dict(saved['state_dict']);model.to('cuda:1').eval();parts=[]
   with torch.no_grad():
    for i in range(0,len(X),2048):
     x=torch.tensor((X[i:i+2048]-saved['feature_mean'])/saved['feature_std'],device='cuda:1');p=model(x)
     parts.append((p[:,:,1] if isq else p).cpu().numpy())
   delta=np.concatenate(parts)*saved['target_std']+saved['target_mean'];del model;torch.cuda.empty_cache()
  preds[(name,seed)]=delta+X[:,-1,0,None];print(name,seed,'scored',len(X),flush=True)
 preds[('Stack',seed)]=np.column_stack([np.column_stack([preds[(m,seed)][:,j] for m in freeze['members']])@np.asarray(freeze['weights'][str(seed)][str(h)]) for j,h in enumerate([30,60,90])])
for name in list(BASE)+['Stack']:
 preds[(name,'mean')]=np.mean([preds[(name,s)] for s in SEEDS],axis=0)
preds[('Persistence','mean')]=np.repeat(X[:,-1,0,None],3,axis=1)
preds[('LinearTrend','mean')]=X[:,-1,0,None]+(X[:,-1,0]-X[:,-2,0])[:,None]*np.array([2,4,6])
for (name,seed),p in preds.items():
 assert np.isfinite(p).all();f=meta.copy()
 for j,h in enumerate([30,60,90]):f[f'true_{h}']=Y[:,j];f[f'pred_{h}']=p[:,j]
 f['dataset']='azt1d';f['model']=name;f['condition']='glucose_only';f['seed']=str(seed);rows.append(f)
pd.concat(rows,ignore_index=True).to_parquet(out/'all_predictions.parquet',index=False)
(out/'complete.json').write_text(json.dumps({'n_patients':meta.PtID.nunique(),'n_origins':len(meta),'predictions_sha256':sha256(out/'all_predictions.parquet'),'freeze_sha256':sha256(out/'freeze.json'),'array_sha256':sha256(data/'azt1d.npz')},indent=2))
