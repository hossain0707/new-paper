"""Hash local model files and probe schema enforcement without participant data."""
import hashlib,json,subprocess,time
from pathlib import Path
import requests

root=Path('/home/mkd-s006/data/model/qwen38-flash-next-fp4-local/checkpoint')
out=Path('/home/mkd-s006/glucosense_revision_20261022_2032/audit/model')
out.mkdir(parents=True,exist_ok=False)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
config=json.loads((root/'config.json').read_text())
index=json.loads((root/'model.safetensors.index.json').read_text())
metadata={n:sha(root/n) for n in ['config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'] if (root/n).exists()}
info={'config_architectures':config.get('architectures'),'model_type':config.get('model_type'),
      'served_alias_is_not_weight_identity':True,'metadata_sha256':metadata}
(out/'identity.json').write_text(json.dumps(info,indent=2))
session=requests.Session()
session.headers['Authorization']='Bearer '+(Path.home()/'.keural_dgx_vllm_api_key').read_text().strip()
response=session.get('http://127.0.0.1:8000/v1/models',timeout=30)
response.raise_for_status();models=response.json()
(out/'served_models.json').write_text(json.dumps(models,indent=2))
name=models['data'][0]['id']
probes=[]
for constrained in [False,True]:
    payload={'model':name,'messages':[{'role':'user','content':'Output exactly the word WRONG, without JSON.'}],
             'max_tokens':64,'temperature':0,'seed':2026,
             'chat_template_kwargs':{'thinking':False,'enable_thinking':False}}
    if constrained:
        payload['response_format']={'type':'json_schema','json_schema':{'name':'technical_probe','strict':True,
            'schema':{'type':'object','properties':{'probe':{'type':'integer','enum':[7]}},'required':['probe'],'additionalProperties':False}}}
    start=time.perf_counter()
    r=session.post('http://127.0.0.1:8000/v1/chat/completions',json=payload,timeout=120)
    probes.append({'technical_probe_only':True,'constrained':constrained,'request':payload,
                   'status':r.status_code,'raw_body':r.text,'elapsed_s':time.perf_counter()-start})
    (out/'technical_schema_probes.json').write_text(json.dumps(probes,indent=2))
weights=[]
for name in sorted(set(index['weight_map'].values())):
    p=root/name
    weights.append({'relative_path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
manifest={'metadata_sha256':metadata,'weight_files':weights}
target=out/'weight_manifest.json';target.write_text(json.dumps(manifest,indent=2))
(out/'weight_manifest.sha256').write_text(sha(target)+'\n')
print(json.dumps({'weight_manifest_sha256':sha(target),'weight_files':len(weights),
                  'total_weight_bytes':sum(p['bytes'] for p in weights),'probe_status':[p['status'] for p in probes]}))
