"""Sequential warm processing replay; includes actual prediction and TreeSHAP."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib,copy,platform,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib,shap,requests
from src.revision.grounding import SYSTEM_PROMPT,expected_output,deliver,study_score
from src.revision.data import sha256
out=ROOT/'results/revision_v3/latency';out.mkdir(parents=True,exist_ok=True);responses=out/'responses';responses.mkdir(exist_ok=True)
old=ROOT/'results/revision_v2/llm';frozen=json.loads((old/'prompt_freeze.json').read_text());assert frozen['detector_sha256']==sha256(ROOT/'src/revision/grounding.py')
cases=[json.loads(p.read_text()) for p in (old/'cases').glob('*.json')];byid={c['case_id']:c for c in cases};examples=[byid[i] for i in frozen['example_ids']]
def source(c):return 'DDATSHR' if c['dataset']=='ddatshr' else ('HUPA' if 'HUPA' in c['PtID'] else 'D1NAMO')
ordered=sorted([c for c in cases if c['split']=='test'],key=lambda c:hashlib.sha256(('v3_latency|'+c['case_id']).encode()).hexdigest())
selected=[]
for s in sorted({source(c) for c in ordered}):selected.append(next(c for c in ordered if source(c)==s))
selected+= [c for c in ordered if c not in selected][:30-len(selected)]
freeze={'case_ids':[c['case_id'] for c in selected],'source_counts':pd.Series([source(c) for c in selected]).value_counts().to_dict(),'script_sha256':sha256(Path(__file__)),'prompt_freeze_sha256':sha256(old/'prompt_freeze.json'),'concurrency':1,'timeout_s':600,'deadline_s':900,'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True),'cpu':platform.processor(),'scope':'available in-memory history through rendering; excludes model loading, device/network acquisition, startup and clinical workflow','server':'TP4, GPU memory utilization .55, fp8 KV cache, eager, max_len8192, max_seqs8; shared server process but no other requests during replay'}
fp=out/'freeze.json'
if fp.exists():assert json.loads(fp.read_text())==freeze
else:fp.write_text(json.dumps(freeze,indent=2))
arrays={d:np.load(ROOT/f'data/processed/revision_v2/history_120/{d}.npz')['X'] for d in ['ddatshr','diadata']}
models={}
for c in selected:
 key=(c['dataset'],c['PtID'])
 if key not in models:
  p=ROOT/'results/revision_v2/fits'/key[0]/'cgm_hr/RandomForest/seed_42'/key[1]/'checkpoint.joblib'
  assert sha256(p)==c['checkpoint_sha256'];saved=joblib.load(p);models[key]=(saved,shap.TreeExplainer(saved['model']))
def run(original,arm,warmup=False):
 path=responses/f"{original['case_id']}_{arm}{'_warmup' if warmup else ''}.json"
 if path.exists():return
 c=copy.deepcopy(original);saved,explainer=models[(c['dataset'],c['PtID'])];start=time.perf_counter()
 x=arrays[c['dataset']][c['array_index']][None,:,:].copy();z=saved['scaler'].transform(x.reshape(1,-1));delta=saved['model'].predict(z);t1=time.perf_counter()
 values=np.asarray(explainer.shap_values(z,check_additivity=True));np.testing.assert_allclose(values.sum(axis=1)+explainer.expected_value,delta,atol=1e-4)
 entries=[]
 for i in np.argsort(np.abs(values[0,:,0]))[-3:][::-1]:
  lag=(8-i//4)*15+(15 if c['dataset']=='diadata' else 0);channel=['glucose_mgdl','hr_bpm','tod_sin','tod_cos'][i%4]
  entries.append({'feature':f'{channel}_lag_{lag}min','contribution_mgdl':round(float(values[0,i,0]),1)})
 glucose=round(float(x[0,-1,0]),1);rate=round(float((x[0,-1,0]-x[0,-2,0])/15),2)
 c['values']={'observations':{'glucose_mgdl':glucose,'hr_bpm':round(float(x[0,-1,1]),1),'input_lag_minutes':15 if c['dataset']=='diadata' else 0},'forecasts_mgdl':{f'h{h}':round(float(delta[0,j]+x[0,-1,0]),1) for j,h in enumerate([30,60,90])},'rate_mgdl_per_min':rate,'study_score':study_score(glucose,rate),'shap_h30':entries}
 assert c['values']==original['values'];t2=time.perf_counter()
 messages=[{'role':'system','content':SYSTEM_PROMPT}]
 if arm=='few_shot':
  for ex in examples:messages.extend([{'role':'user','content':json.dumps({'values':ex['values']})},{'role':'assistant','content':json.dumps(expected_output(ex))}])
 messages.append({'role':'user','content':json.dumps({'values':c['values']})})
 payload={'model':frozen['model'],'messages':messages,'temperature':0,'seed':2026,'max_tokens':2048,'chat_template_kwargs':{'thinking':False,'enable_thinking':False},'reasoning_effort':'none'}
 text='';raw=None;error=None
 try:
  if arm=='deterministic':text=json.dumps(expected_output(c))
  else:
   r=requests.post('http://127.0.0.1:8002/v1/chat/completions',json=payload,timeout=600);r.raise_for_status();raw=r.json();text=raw['choices'][0]['message'].get('content') or ''
 except Exception as e:error=str(e)
 t3=time.perf_counter();result=deliver(text,c);t4=time.perf_counter()
 if arm=='deterministic':result['origin']='deterministic_template'
 record={'case_id':c['case_id'],'dataset':c['dataset'],'PtID':c['PtID'],'source':source(c),'arm':arm,'warmup':warmup,'prediction_s':t1-start,'shap_s':t2-t1,'generation_s':t3-t2,'validation_render_s':t4-t3,'total_s':t4-start,'deadline_missed':t4-start>900,'error':error,'raw_response':raw,'raw_text':text,**result}
 path.write_text(json.dumps(record,indent=2));print(c['case_id'],arm,round(t4-start,3),result['validation']['accepted'],flush=True)
for arm in ['deterministic','zero_shot','few_shot']:run(selected[0],arm,True)
# Rotate arm order to reduce systematic position effects, keeping concurrency one.
for i,c in enumerate(selected):
 arms=['deterministic','zero_shot','few_shot'];arms=arms[i%3:]+arms[:i%3]
 for arm in arms:run(c,arm)
rows=[]
for p in responses.glob('*.json'):
 r=json.loads(p.read_text())
 if r['warmup']:continue
 rows.append({k:r[k] for k in ['case_id','dataset','PtID','source','arm','prediction_s','shap_s','generation_s','validation_render_s','total_s','deadline_missed','error']}|{'accepted':r['validation']['accepted']})
f=pd.DataFrame(rows);assert len(f)==90;f.to_csv(out/'measurements.csv',index=False);summary=[]
for arm,g in f.groupby('arm'):
 for component in ['prediction_s','shap_s','generation_s','validation_render_s','total_s']:
  summary.append(dict(arm=arm,component=component,n=len(g),median=g[component].median(),p95=g[component].quantile(.95),maximum=g[component].max(),accepted=int(g.accepted.sum()),deadline_missed=int(g.deadline_missed.sum()),request_errors=int(g.error.notna().sum())))
pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
print('COMPLETE',len(f),flush=True)
