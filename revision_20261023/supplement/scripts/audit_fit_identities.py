import sys,json,hashlib
from pathlib import Path
import pandas as pd
root=Path(sys.argv[1]);files=list((root/'results/revision_v4/stack_full/validation/fits').rglob('complete.json'));assert len(files)==690
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={'code_sha256':sha(root/'src/revision/strengthening.py'),'base_code_sha256':sha(root/'src/revision/experiment.py'),'deep_code_sha256':sha(root/'src/models/deep_models.py'),'protocol_sha256':sha(root/'protocol_v3.md')}
for f in files:
 m=json.loads(f.read_text());i=m['identity'];cohort=f.relative_to(root/'results/revision_v4/stack_full/validation/fits').parts[0]
 for k,v in expected.items():assert i[k]==v,(f,k)
 assert i['meta_sha256']==sha(root/f'data/processed/revision_v2/history_120/{cohort}_samples.parquet')
 if i['model'] in ['RandomForest','XGBoost']:
  s=pd.read_csv(f.parent/'search.csv');best=s.loc[s.validation_macro_MAE.idxmin()]
  for k,v in m['config'].items():assert abs(float(best[k])-v)<1e-8,(f,k)
 else:
  s=pd.read_csv(f.parent/'training_history.csv');best=s.loc[s.validation_MAE.idxmin()]
  assert abs(float(best.validation_MAE)-m['config']['inner_MAE'])<1e-6
  assert float(best.lr)==m['config']['lr'] and int(best.epoch)==m['config']['epoch']
print(json.dumps({'fit_manifests':len(files),'all_source_protocol_and_metadata_hashes_match':True,'all_690_configurations_match_best_validation_score':True,'source_hashes':expected},indent=2))
