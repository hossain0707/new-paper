"""Check saved baseline horizons and independence from other inference contexts."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from chronos import Chronos2Pipeline
from huggingface_hub import snapshot_download
from src.revision.data import sha256
out=ROOT/'results/revision_v3/chronos';freeze=json.loads((out/'freeze.json').read_text());assert freeze['script_sha256']==sha256(ROOT/'scripts/run_revision_chronos.py')
pipeline=Chronos2Pipeline.from_pretrained(freeze['model'],revision=freeze['revision'],device_map='cuda:0');checks=[]
for dataset,steps,offsets in [('ddatshr',6,[1,3,5]),('diadata',7,[2,4,6])]:
 arr=np.load(ROOT/f'data/processed/revision_v2/history_120/{dataset}.npz');meta=pd.read_parquet(ROOT/f'data/processed/revision_v2/history_120/{dataset}_samples.parquet');saved=pd.read_parquet(out/f'{dataset}_predictions.parquet')
 assert saved.sample_id.equals(meta.sample_id);np.testing.assert_array_equal(saved[[f'true_{h}' for h in [30,60,90]]],arr['y'])
 x=arr['X'][:256,:,0][:,None,:];q,_=pipeline.predict_quantiles(x,prediction_length=steps,quantile_levels=[.5],batch_size=256,cross_learning=False)
 pred=np.stack([t.cpu().numpy()[0,:,0] for t in q])[:,offsets];np.testing.assert_allclose(pred,saved.iloc[:256][[f'pred_{h}' for h in [30,60,90]]],atol=1e-3,rtol=1e-5)
 changed=x.copy();changed[1:]=400
 alt,_=pipeline.predict_quantiles(changed,prediction_length=steps,quantile_levels=[.5],batch_size=256,cross_learning=False)
 np.testing.assert_allclose(q[0].cpu(),alt[0].cpu(),atol=1e-4,rtol=1e-6)
 checks.append(f'{dataset}: all saved origins/targets match, first 256 forecasts reproduce, changing other contexts does not change first forecast')
modelpath=Path(snapshot_download(freeze['model'],revision=freeze['revision'],local_files_only=True))
assets={str(p.relative_to(modelpath)):sha256(p) for p in modelpath.rglob('*') if p.is_file()}
(out/'verification.json').write_text(json.dumps({'status':'passed','checks':checks,'predictions_sha256':{d:sha256(out/f'{d}_predictions.parquet') for d in ['ddatshr','diadata']},'cached_model_files':assets,'model_dtype':str(pipeline.model.dtype),'freeze_sha256':sha256(out/'freeze.json')},indent=2));print('\n'.join(checks))
