"""Freeze development-only model selection before any AZT1D scoring."""
import sys,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.revision.strengthening import weighted_simplex
from src.revision.experiment import BASE,SEEDS
from src.revision.reporting import STACK
from src.revision.data import sha256
out=ROOT/'results/revision_v3/external';target=out/'freeze.json'
assert not target.exists(), 'Preserve existing freeze; do not overwrite'
files={};weights={}
for seed in SEEDS:
 frames={}
 for model in BASE:
  p=out/'fits/ddatshr/glucose_only'/model/f'seed_{seed}/development'
  manifest=json.loads((p/'complete.json').read_text())
  assert manifest['test_n']==0 and not manifest['identity']['test']
  for f in p.iterdir():files[str(f.relative_to(ROOT))]=sha256(f)
  frames[model]=pd.read_parquet(p/'validation.parquet').sort_values('sample_id').reset_index(drop=True)
 ref=frames[STACK[0]]
 weights[str(seed)]={}
 for h in (30,60,90):
  for m in STACK:
   assert frames[m].sample_id.equals(ref.sample_id)
   np.testing.assert_array_equal(frames[m][f'true_{h}'],ref[f'true_{h}'])
  pred=np.column_stack([frames[m][f'pred_{h}'] for m in STACK])/100
  weights[str(seed)][str(h)]=weighted_simplex(pred,ref[f'true_{h}'].to_numpy()/100,ref.PtID.to_numpy()).tolist()
for name in ['protocol_v3.md','src/revision/strengthening.py','src/revision/experiment.py','scripts/freeze_revision_external.py']:
 files[name]=sha256(ROOT/name)
target.write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'members':STACK,'weights':weights,'files':files,'external_scoring_started':False},indent=2))
print('Frozen',len(files),'artifacts',sha256(target))
