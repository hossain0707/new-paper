"""Sequential local factorial comparison, with immutable configuration and raw logs.

Use development first. This is a post-review analysis, not preregistered confirmation.
JSON-schema requests must be supported by the server; failures are never retried as
unconstrained generation. Does not overwrite any v2/v3 results.
"""
import argparse,csv,datetime,hashlib,itertools,json,platform,sys,time
from pathlib import Path
from urllib.parse import urlparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import requests
from src.revision.comparison import contract_schema,messages_for
from src.revision.grounding import deliver,expected_output

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,help='Exact served model identifier')
    p.add_argument('--checkpoint-id',required=True,help='Verified checkpoint revision/hash, not just served alias')
    p.add_argument('--base-url',default='http://127.0.0.1:8000/v1')
    p.add_argument('--split',choices=['development','test'],default='development')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--api-key-file',type=Path,help='Local secret file; contents are never saved')
    p.add_argument('--case-source',type=Path,default=ROOT/'results/revision_v2/llm')
    a=p.parse_args()
    if urlparse(a.base_url).hostname not in ('127.0.0.1','localhost','::1'):
        raise ValueError('Only local loopback inference is allowed')
    source=a.case_source;selection=source/'case_selection.csv'
    with selection.open() as f: rows=list(csv.DictReader(f))
    cases=[]
    for row in rows:
        matches=list((source/'cases').glob('*.json'))
        if not matches: raise RuntimeError('Raw frozen cases are missing; rebuild from verified RF predictions first')
        break
    for path in sorted((source/'cases').glob('*.json')):
        cases.append((path,json.loads(path.read_text())))
    bysample={c['sample_id']:c for _,c in cases}
    required={r['sample_id'] for r in rows if r['split']==a.split}
    if not required or not required<=set(bysample): raise RuntimeError('Selected cases incomplete')
    dev={r['PtID'] for r in rows if r['split']=='development'}
    test={r['PtID'] for r in rows if r['split']=='test'}
    if dev & test: raise RuntimeError('Development/test participant overlap')
    freeze=json.loads((source/'prompt_freeze.json').read_text())
    if freeze['selection_sha256']!=digest(selection):
        raise RuntimeError('Case selection differs from the historical freeze')
    bycase={c['case_id']:c for _,c in cases}
    examples=[bycase[s] for s in freeze['example_ids']]
    if any(c['split']!='development' for c in examples): raise RuntimeError('Nondevelopment example')
    session=requests.Session()
    if a.api_key_file:
        session.headers['Authorization']='Bearer '+a.api_key_file.read_text().strip()
    available=session.get(a.base_url.rstrip('/')+'/models',timeout=30);available.raise_for_status()
    models=available.json()
    if a.model not in {m['id'] for m in models['data']}: raise RuntimeError('Requested model is not served')
    config={'analysis':'post-review factorial; no claim of clinical evaluation',
      'model':a.model,'checkpoint_id':a.checkpoint_id,'base_url':a.base_url,'split':a.split,
      'temperature':0,'seed':2026,'max_tokens':a.max_tokens,'execution':'sequential client requests',
      'schema_api':'OpenAI response_format json_schema strict; backend must enforce',
      'protocol_v4_sha256':digest(ROOT/'protocol_v4.md'),
      'selection_sha256':digest(selection),'examples':freeze['example_ids'],
      'cases_sha256':{path.name:digest(path) for path,_ in cases},
      'code_sha256':{str(path.relative_to(ROOT)):digest(path) for path in
       [Path(__file__),ROOT/'src/revision/comparison.py',ROOT/'src/revision/grounding.py']}}
    a.output.mkdir(parents=True,exist_ok=True);cp=a.output/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=config: raise RuntimeError('Output configuration differs; use a new directory')
    cp.write_text(json.dumps(config,indent=2))
    (a.output/'server_models.json').write_text(json.dumps(models,indent=2))
    (a.output/'client_environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'requests':requests.__version__},indent=2))
    results=[]
    arms=[('deterministic',False,False,False)]+[
       (f'{"few" if few else "zero"}_{"supplied" if supplied else "calculated"}_{"schema" if schema else "plain"}',few,supplied,schema)
       for few,supplied,schema in itertools.product([False,True],repeat=3)]
    for case in sorted((bysample[s] for s in required),key=lambda c:c['case_id']):
        case_id=case['case_id']
        if Path(case_id).name!=case_id: raise ValueError('Unsafe case identifier')
        # Rotate order by case to distribute warm-up/drift effects across conditions.
        shift=int(hashlib.sha256(case_id.encode()).hexdigest()[:8],16)%len(arms)
        for arm,few,supplied,schema in arms[shift:]+arms[:shift]:
            path=a.output/f'{case_id}_{arm}.json'
            if path.exists(): results.append(json.loads(path.read_text()));continue
            payload={'model':a.model,'messages':messages_for(case,examples,few,supplied),
                     'temperature':0,'seed':2026,'max_tokens':a.max_tokens,
                     'chat_template_kwargs':{'thinking':False,'enable_thinking':False}}
            if schema: payload['response_format']={'type':'json_schema','json_schema':{
                'name':'grounded_output','strict':True,'schema':contract_schema(case)}}
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
            start=time.perf_counter();raw=None;error=None;text='';status=None;raw_http_body=None
            try:
                if arm=='deterministic': text=json.dumps(expected_output(case))
                else:
                    response=session.post(a.base_url.rstrip('/')+'/chat/completions',json=payload,timeout=600)
                    status=response.status_code;raw_http_body=response.text
                    response.raise_for_status();raw=response.json();text=raw['choices'][0]['message'].get('content') or ''
            except Exception as exc: error=str(exc)
            elapsed=time.perf_counter()-start
            validation_start=time.perf_counter();delivered=deliver(text,case)
            validation_render_s=time.perf_counter()-validation_start
            result={'case_id':case_id,'PtID':case['PtID'],'dataset':case['dataset'],'split':a.split,
              'arm':arm,'latency_s':elapsed,'error':error,'raw_text':text,'raw_response':raw,
              'request_started_utc':started_utc,
              'http_status':status,'raw_http_body':raw_http_body,
              'validation_render_s':validation_render_s,'total_contract_pipeline_s':time.perf_counter()-start,
              'request':None if arm=='deterministic' else payload,**delivered}
            if arm=='deterministic':result['origin']='deterministic_template'
            path.write_text(json.dumps(result,indent=2));results.append(result)
            print(case_id,arm,'ERROR' if error else result['validation']['accepted'],flush=True)
    fields=['arm','n','accepted','errors','median_client_request_s']
    import statistics
    with (a.output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for arm,*_ in arms:
            group=[r for r in results if r['arm']==arm]
            writer.writerow({'arm':arm,'n':len(group),'accepted':sum(r['validation']['accepted'] and not r['error'] for r in group),
               'errors':sum(bool(r['error']) for r in group),'median_client_request_s':statistics.median(r['latency_s'] for r in group)})

if __name__=='__main__':main()
