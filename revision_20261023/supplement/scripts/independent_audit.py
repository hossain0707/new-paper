"""Read-only independent verification of frozen raw artifacts; writes a new audit only."""
import sys,json,hashlib,itertools,csv
from pathlib import Path
from collections import Counter
ROOT=Path(sys.argv[1]); OUT=Path(sys.argv[2]); OUT.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.revision.grounding import validate,expected_output
from src.revision.comparison import messages_for,contract_schema
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v4=ROOT/'results/revision_v4'; casesroot=v4/'recovered_cases'
cases={x.stem:json.loads(x.read_text()) for x in (casesroot/'cases').glob('*.json')}
selection=list(csv.DictReader((casesroot/'case_selection.csv').open()))
freeze=json.loads((casesroot/'prompt_freeze.json').read_text())
assert sha(casesroot/'case_selection.csv')==freeze['selection_sha256']
dev={x['PtID'] for x in selection if x['split']=='development'}
test={x['PtID'] for x in selection if x['split']=='test'}
assert len(dev)==8 and len(test)==38 and not dev&test
examples=[cases[x] for x in freeze['example_ids']]
assert all(x['split']=='development' for x in examples)
qreport={};rawhash={}
for split,n in [('development',24),('test',111)]:
 folder=v4/'spark_20261022/qwen'/split; config=json.loads((folder/'configuration.json').read_text())
 assert config['code_sha256']['src/revision/grounding.py']==sha(ROOT/'src/revision/grounding.py')
 assert config['code_sha256']['src/revision/comparison.py']==sha(ROOT/'src/revision/comparison.py')
 assert config['selection_sha256']==sha(casesroot/'case_selection.csv')
 ids={k for k,c in cases.items() if c['split']==split}; assert len(ids)==n
 arms=['deterministic']+[f'{"few" if f else "zero"}_{"supplied" if d else "calculated"}_{"schema" if s else "plain"}' for f,d,s in itertools.product([False,True],repeat=3)]
 rows=[]
 for cid,arm in itertools.product(sorted(ids),arms):
  p=folder/f'{cid}_{arm}.json'; r=json.loads(p.read_text()); c=cases[cid]
  assert r['case_id']==cid and r['PtID']==c['PtID'] and r['split']==split
  check=validate(r['raw_text'],c); assert check==r['validation'],str(p)
  rawhash[str(p.relative_to(ROOT))]=sha(p)
  if arm!='deterministic':
   few=arm.startswith('few'); supplied='_supplied_' in arm; schema=arm.endswith('_schema')
   request=r['request']; assert request['messages']==messages_for(c,examples,few,supplied)
   assert request['temperature']==0 and request['seed']==2026 and request['max_tokens']==2048
   assert ('response_format' in request)==schema
   if schema:assert request['response_format']['json_schema']['schema']==contract_schema(c)
   assert r['raw_response']==json.loads(r['raw_http_body']) and r['http_status']==200
   assert r['raw_text']==r['raw_response']['choices'][0]['message']['content']
  rows.append({'case_id':cid,'participant':c['PtID'],'arm':arm,'accepted':int(check['accepted'] and not r['error']),'seconds':r['latency_s'],'error':bool(r['error'])})
 frame=pd.DataFrame(rows); summary=frame.groupby('arm').agg(n=('accepted','size'),accepted=('accepted','sum'),median_request_s=('seconds','median'),errors=('error','sum')).reset_index()
 old=pd.read_csv(v4/f'spark_20261022/qwen/{split}_summary/arm_summary.csv').set_index('arm')
 for r in summary.to_dict('records'):
  assert r['n']==int(old.loc[r['arm'],'n_cases']) and r['accepted']==int(old.loc[r['arm'],'accepted'])
  assert abs(r['median_request_s']-old.loc[r['arm'],'median_request_s'])<1e-10
 summary.to_csv(OUT/f'qwen_{split}_independent.csv',index=False)
 qreport[split]={'cases':n,'LLM_requests':n*8,'deterministic':n,'raw_validation_matches':len(rows),'prompt_schema_matches':n*8,'summary_matches':True}
npz={};meta={}; fitrows=[]; checkrows=[]; weightrows=[]
members=['RandomForest','XGBoost','TFTEncoder','BiLSTMTransformer','CNNGRU']
for dataset in ['ddatshr','diadata']:
 data=ROOT/'data/processed/revision_v2/history_120'; npz[dataset]=np.load(data/f'{dataset}.npz')
 meta[dataset]=pd.read_parquet(data/f'{dataset}_samples.parquet');meta[dataset].PtID=meta[dataset].PtID.astype(str)
base=v4/'stack_full/validation/fits'
files=list(base.rglob('complete.json')); assert len(files)==690
for f in sorted(files):
 m=json.loads(f.read_text()); i=m['identity']; dataset=f.relative_to(base).parts[0]; person=i['test'][0]
 people=sorted(set(meta[dataset].PtID)-{person});rng=np.random.default_rng(712+int(hashlib.sha256(person.encode()).hexdigest()[:8],16));rng.shuffle(people)
 assert i['train']==people[3:] and i['validation']==people[:3] and len(i['test'])==1
 assert not set(i['train'])&set(i['validation']) and not set(i['train'])&set(i['test']) and not set(i['validation'])&set(i['test'])
 assert i['data_sha256']==sha(data/f'{dataset}.npz') if dataset=='diadata' else i['data_sha256']==sha(ROOT/f'data/processed/revision_v2/history_120/{dataset}.npz')
 folder=f.parent;val=pd.read_parquet(folder/'validation.parquet');testframe=pd.read_parquet(folder/'test.parquet')
 assert set(val.PtID.astype(str))==set(i['validation']) and set(testframe.PtID.astype(str))=={person}
 assert len(val)==m['validation_n'] and len(testframe)==m['test_n']
 fitrows.append({'path':str(f.relative_to(ROOT)),'sha256':sha(f),'dataset':dataset,'model':i['model'],'seed':i['seed'],'split_verified':True})
 # Predetermined representative: first held-out person, seed 42, all five members, both cohorts.
 if person==sorted(meta[dataset].PtID.unique())[0] and i['seed']==42:
  X=npz[dataset]['X'];mask=meta[dataset].PtID.eq(person).to_numpy();x=X[mask][:16];truth=testframe.iloc[:16]
  if i['model'] in members[:2]:
   import joblib
   ck=folder/'checkpoint.joblib';b=joblib.load(ck);pred=b['model'].predict(b['scaler'].transform(x.reshape(len(x),-1)))+x[:,-1,0,None]
  else:
   import torch
   from src.revision.experiment import make_deep
   torch.set_num_threads(1);ck=folder/'checkpoint.pt';b=torch.load(ck,map_location='cpu',weights_only=False)
   from src.revision.experiment import seed_all
   seed_all(42)
   model,isq=make_deep(i['model'],x.shape[-1]);model.load_state_dict(b['state_dict']);model.to('cuda:0').eval()
   # Match the original GPU device and 2048-example evaluation batch boundaries.
   fullx=X[mask];pieces=[]
   with torch.no_grad():
    for start in range(0,len(fullx),2048):
     z=model(torch.tensor((fullx[start:start+2048]-b['feature_mean'])/b['feature_std'],device='cuda:0'));pieces.append((z[:,:,1] if isq else z).cpu().numpy())
   pred=np.concatenate(pieces)[:16]*b['target_std']+b['target_mean']+x[:,-1,0,None]
  diff=float(np.max(np.abs(pred-truth[['pred_30','pred_60','pred_90']].to_numpy())))
  assert diff<.002,(str(ck),diff)
  checkrows.append({'dataset':dataset,'model':i['model'],'seed':42,'samples':len(x),'max_abs_mgdl':diff,'checkpoint_sha256':sha(ck)})
for wfile in sorted((v4/'stack_full/validation/stack').rglob('weights.json')):
 rel=wfile.relative_to(v4/'stack_full/validation/stack');dataset,seed,person=rel.parts[:3]
 w=json.loads(wfile.read_text());assert w['members']==members
 frames={model:{s:pd.read_parquet(base/dataset/'cgm_hr'/model/seed/person/f'{s}.parquet').sort_values('sample_id').reset_index(drop=True) for s in ['validation','test']} for model in members}
 reference=frames[members[0]];stack=pd.read_parquet(wfile.parent/'test.parquet').sort_values('sample_id').reset_index(drop=True)
 for h in [30,60,90]:
  for model in members:
   for s in ['validation','test']:assert frames[model][s].sample_id.equals(reference[s].sample_id)
  v=reference['validation'];P=np.column_stack([frames[m]['validation'][f'pred_{h}'] for m in members])/100;y=v[f'true_{h}'].to_numpy()/100
  counts=v.PtID.value_counts();ww=1/v.PtID.map(counts).to_numpy();ww=ww/ww.mean()
  opt=minimize(lambda x:np.mean(ww*(P@x-y)**2),np.ones(5)/5,jac=lambda x:2*P.T@(ww*(P@x-y))/len(y),bounds=[(0,1)]*5,constraints={'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones(5)},method='SLSQP',options={'maxiter':1000,'ftol':1e-9})
  assert opt.success;assert np.max(np.abs(opt.x-w['weights'][str(h)]))<1e-6
  pred=np.column_stack([frames[m]['test'][f'pred_{h}'] for m in members])@opt.x
  assert np.max(np.abs(pred-stack[f'pred_{h}']))<1e-5
 weightrows.append({'weights_path':str(wfile.relative_to(ROOT)),'sha256':sha(wfile),'validation_only_refit':True,'held_out_prediction_match':True})
assert len(checkrows)==10 and len(weightrows)==138
pd.DataFrame(fitrows).to_csv(OUT/'stack_690_manifest_audit.csv',index=False)
pd.DataFrame(checkrows).to_csv(OUT/'representative_checkpoint_audit.csv',index=False)
pd.DataFrame(weightrows).to_csv(OUT/'stack_138_weight_audit.csv',index=False)
(OUT/'raw_response_hashes.json').write_text(json.dumps(rawhash,indent=2))
(OUT/'VERIFIED.json').write_text(json.dumps({'Qwen':qreport,'participant_overlap':0,'examples_development_only':True,'frozen_validator_sha256':sha(ROOT/'src/revision/grounding.py'),'stack_fit_manifests':690,'validation_only_weight_sets':138,'representative_checkpoints':10,'historical_DeepSeek_reproduced':False},indent=2))
print((OUT/'VERIFIED.json').read_text())
