"""Local-only controlled generation; raw outputs and failures are retained verbatim."""
import argparse,concurrent.futures,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import requests
import pandas as pd
from src.revision.grounding import SYSTEM_PROMPT,expected_output,deliver,render
from src.revision.data import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','test'],default='development')
    p.add_argument('--endpoint',default='http://127.0.0.1:8002/v1/chat/completions');p.add_argument('--workers',type=int,default=8)
    p.add_argument('--available',action='store_true',help='Run only already-built, preselected cases; final reporting still requires the full set')
    a=p.parse_args()
    if not a.endpoint.startswith(('http://127.0.0.1:','http://localhost:')):raise ValueError('This runner only permits loopback inference')
    out=ROOT/'results/revision_v2/llm';all_cases=[json.loads(p.read_text()) for p in sorted((out/'cases').glob('*.json'))]
    selected=pd.read_csv(out/'case_selection.csv');available={c['sample_id'] for c in all_cases}
    required=set(selected[selected.split.eq(a.split)].sample_id)
    if not a.available and not required<=available:raise RuntimeError(f'{len(required-available)} selected cases still lack completed RF fits')
    dev=[c for c in all_cases if c['split']=='development'];examples=[]
    for state in ['low','in_range','high']:
        chosen=selected[(selected.split=='development')&(selected.state==state)].sort_values(['dataset','PtID']).iloc[0]
        candidates=[c for c in dev if c['sample_id']==chosen.sample_id]
        if not candidates:raise RuntimeError('The predetermined development example is not built yet')
        examples.append(candidates[0])
    freeze={'system_prompt':SYSTEM_PROMPT,'example_ids':[c['case_id'] for c in examples],
            'detector_sha256':sha256(ROOT/'src/revision/grounding.py'),
            'selection_sha256':sha256(out/'case_selection.csv'),'model':'deepseek-ai/DeepSeek-V4-Flash-0731',
            'model_revision':'7872f01b1d1fe23eabc4c98b48bffcef5a386062','temperature':0,'seed':2026,
            'max_tokens':2048,'thinking':False,'reasoning_effort':'none'}
    frozen=out/'prompt_freeze.json'
    if frozen.exists() and json.loads(frozen.read_text())!=freeze:raise RuntimeError('Frozen prompt/config changed')
    frozen.write_text(json.dumps(freeze,indent=2))
    responses=out/'responses';responses.mkdir(exist_ok=True)
    def run(case,arm):
        path=responses/f"{case['case_id']}_{arm}.json"
        if path.exists():return json.loads(path.read_text())
        messages=[{'role':'system','content':SYSTEM_PROMPT}]
        if arm=='few_shot':
            for ex in examples:
                messages.extend([{'role':'user','content':json.dumps({'values':ex['values']})},
                                 {'role':'assistant','content':json.dumps(expected_output(ex))}])
        messages.append({'role':'user','content':json.dumps({'values':case['values']})})
        payload={'model':freeze['model'],'messages':messages,'temperature':0,'seed':2026,'max_tokens':2048,
                 'chat_template_kwargs':{'thinking':False,'enable_thinking':False},'reasoning_effort':'none'}
        start=time.monotonic();raw=None;error=None;text=''
        try:
            if arm=='deterministic':text=json.dumps(expected_output(case))
            else:
                r=requests.post(a.endpoint,json=payload,timeout=600);r.raise_for_status();raw=r.json()
                text=raw['choices'][0]['message'].get('content') or ''
        except Exception as e:error=str(e)
        elapsed=time.monotonic()-start;delivered=deliver(text,case)
        if arm=='deterministic':delivered['origin']='deterministic_template'
        result={'case_id':case['case_id'],'dataset':case['dataset'],'PtID':case['PtID'],'split':case['split'],
                'state':case['state'],'arm':arm,'latency_s':elapsed,'error':error,'raw_text':text,'raw_response':raw,
                'request':payload if arm!='deterministic' else None,**delivered}
        path.write_text(json.dumps(result,indent=2));print(case['case_id'],arm,delivered['origin'],round(elapsed,2),flush=True)
        return result
    tasks=[(c,arm) for c in all_cases if c['split']==a.split for arm in ['deterministic','zero_shot','few_shot']]
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run,*t) for t in tasks]):f.result()
    rows=[]
    for path in responses.glob('*.json'):
        r=json.loads(path.read_text());rows.append({k:r[k] for k in ['case_id','dataset','PtID','split','state','arm','latency_s','error','origin']}|
                                                  {'accepted':r['validation']['accepted'],'verified_leaves':r['validation']['verified_leaves']})
    pd.DataFrame(rows).to_csv(out/'generation_metrics.csv',index=False)

if __name__=='__main__':main()
