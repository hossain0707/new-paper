"""Pinned, univariate zero-shot baseline on unchanged primary evaluation origins."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,datetime,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from src.revision.data import sha256
out=ROOT/'results/revision_v3/chronos';out.mkdir(parents=True,exist_ok=True)
revision='29ec3766d36d6f73f0696f85560a422f50e8498c'
freeze={'model':'amazon/chronos-2','revision':revision,'package':importlib.metadata.version('chronos-forecasting'),'transformers':importlib.metadata.version('transformers'),'torch':torch.__version__,'cross_learning':False,'context_length':9,'quantile':.5,'batch_size':256,'script_sha256':sha256(Path(__file__)),'pretraining_overlap':'unknown; not an independently uncontaminated benchmark'}
path=out/'freeze.json'
if path.exists():assert json.loads(path.read_text())==freeze
else:path.write_text(json.dumps(freeze,indent=2))
pipeline=Chronos2Pipeline.from_pretrained('amazon/chronos-2',revision=revision,device_map='cuda:0')
for dataset,steps,indices in [('ddatshr',6,[1,3,5]),('diadata',7,[2,4,6])]:
 target=out/f'{dataset}_predictions.parquet'
 if target.exists():continue
 data=ROOT/'data/processed/revision_v2/history_120';arr=np.load(data/f'{dataset}.npz');X=arr['X'][:,:,0][:,None,:]
 frame=pd.read_parquet(data/f'{dataset}_samples.parquet');parts=[];start=time.perf_counter()
 for offset in range(0,len(X),2048):
  q,_=pipeline.predict_quantiles(X[offset:offset+2048],prediction_length=steps,quantile_levels=[.5],batch_size=256,cross_learning=False)
  # predict_quantiles returns per-task (variates, prediction_length, quantiles).
  chunk=np.stack([t.cpu().numpy()[0,:,0] for t in q])[:,indices]
  assert chunk.shape==(min(2048,len(X)-offset),3) and np.isfinite(chunk).all();parts.append(chunk)
  print(dataset,offset+len(chunk),'/',len(X),flush=True)
 pred=np.concatenate(parts)
 for j,h in enumerate([30,60,90]):frame[f'true_{h}']=arr['y'][:,j];frame[f'pred_{h}']=pred[:,j]
 frame['dataset']=dataset;frame['condition']='glucose_only';frame['model']='Chronos2';frame['seed']='mean';frame.to_parquet(target,index=False)
 (out/f'{dataset}_complete.json').write_text(json.dumps({'n':len(frame),'duration_s':time.perf_counter()-start,'input_sha256':sha256(data/f'{dataset}.npz'),'predictions_sha256':sha256(target),'freeze_sha256':sha256(path)},indent=2))
