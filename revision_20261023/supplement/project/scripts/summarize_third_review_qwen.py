"""Complete-arm Qwen summaries with participant clustering and retained failures."""
import argparse,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.revision.grounding import expected_output

def leaves(v):
    if isinstance(v,dict):return sum(leaves(x) for x in v.values())
    if isinstance(v,list):return sum(leaves(x) for x in v)
    return 1

def interval(a):
    a=np.asarray(a);rng=np.random.default_rng(917)
    b=a[rng.integers(0,len(a),size=(10000,len(a)))].mean(axis=1)
    return float(a.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--case-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((a.input/'configuration.json').read_text())
    selection=pd.read_csv(a.case_source/'case_selection.csv',dtype={'PtID':str})
    selected=selection[selection.split.eq(config['split'])]
    cases={c['case_id']:c for f in (a.case_source/'cases').glob('*.json') for c in [json.loads(f.read_text())]}
    ids={c['case_id'] for c in cases.values() if c['sample_id'] in set(selected.sample_id)}
    arms=['deterministic']+[f'{"few" if f else "zero"}_{"supplied" if d else "calculated"}_{"schema" if s else "plain"}' for f,d,s in itertools.product([False,True],repeat=3)]
    rows=[];raw_hashes={}
    for cid,arm in itertools.product(sorted(ids),arms):
        path=a.input/f'{cid}_{arm}.json'
        r=json.loads(path.read_text());assert r['case_id']==cid and r['arm']==arm
        raw_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        raw=r['raw_response'] or {};choice=(raw.get('choices') or [{}])[0]
        errors=r['validation']['errors']
        rows.append({'case_id':cid,'PtID':str(r['PtID']),'dataset':r['dataset'],'arm':arm,
          'accepted':int(r['validation']['accepted'] and not r['error']),
          'request_error':int(bool(r['error'])),'direction_error':int(any('.direction:' in e for e in errors)),
          'tier_error':int(any('.study_tier:' in e for e in errors)),
          'parse_error':int(any(e.startswith('parse:') for e in errors)),
          'missing_field':int(any(': missing' in e for e in errors)),
          'undeclared_field':int(any(': undeclared field' in e for e in errors)),
          'verified_leaves':r['validation']['verified_leaves'],'expected_leaves':leaves(expected_output(cases[cid])),
          'request_s':r['latency_s'],'contract_pipeline_s':r['total_contract_pipeline_s'],
          'started_utc':r['request_started_utc'],'finish_reason':choice.get('finish_reason'),
          'output_tokens':raw.get('usage',{}).get('completion_tokens')})
    frame=pd.DataFrame(rows);assert len(frame)==len(ids)*9
    assert len(ids)==len(selected)
    frame.to_csv(a.output/'case_arm_metrics.csv',index=False)
    summaries=[]
    for arm,g in frame.groupby('arm'):
        macro=g.groupby('PtID').accepted.mean();mean,lo,hi=interval(macro)
        ordered=g.sort_values('started_utc');steady=ordered.iloc[1:]
        summaries.append({'arm':arm,'n_cases':len(g),'n_people':len(macro),'accepted':int(g.accepted.sum()),
          'acceptance_case':g.accepted.mean(),'acceptance_patient_macro':mean,'macro_ci95_low':lo,'macro_ci95_high':hi,
          'request_errors':int(g.request_error.sum()),'direction_errors':int(g.direction_error.sum()),
          'tier_errors':int(g.tier_error.sum()),'parse_errors':int(g.parse_error.sum()),
          'missing_field_cases':int(g.missing_field.sum()),'undeclared_field_cases':int(g.undeclared_field.sum()),
          'leaf_fidelity_all_expected':g.verified_leaves.sum()/g.expected_leaves.sum(),
          'first_request_s':ordered.request_s.iloc[0],'steady_median_request_s':steady.request_s.median(),
          'median_request_s':g.request_s.median(),'p95_request_s':g.request_s.quantile(.95),
          'median_contract_pipeline_s':g.contract_pipeline_s.median(),'mean_output_tokens':g.output_tokens.mean()})
    pd.DataFrame(summaries).to_csv(a.output/'arm_summary.csv',index=False)
    llm=frame[~frame.arm.eq('deterministic')].copy()
    llm['few']=llm.arm.str.startswith('few_');llm['supplied']=llm.arm.str.contains('_supplied_');llm['schema']=llm.arm.str.endswith('_schema')
    contrasts=[]
    for factor in ['few','supplied','schema']:
        paired=llm.groupby(['PtID',factor]).accepted.mean().unstack()
        d=paired[True]-paired[False];effect,lo,hi=interval(d)
        pvalue=1.0 if np.allclose(d,0) else float(wilcoxon(d,zero_method='pratt',method='auto').pvalue)
        contrasts.append({'factor':factor,'n_people':len(d),'acceptance_effect':effect,'ci95_low':lo,'ci95_high':hi,'p_raw':pvalue})
    pvalues=np.array([r['p_raw'] for r in contrasts]);order=np.argsort(pvalues);adjusted=np.empty(3)
    adjusted[order]=np.minimum(1,np.maximum.accumulate(pvalues[order]*(3-np.arange(3))))
    for r,value in zip(contrasts,adjusted):r['p_holm_exploratory_3']=float(value)
    pd.DataFrame(contrasts).to_csv(a.output/'factor_contrasts.csv',index=False)
    (a.output/'provenance.json').write_text(json.dumps({'post_review':True,'model_configuration':config,'raw_sha256':raw_hashes,
      'resampling_unit':'participant','nboot':10000,'seed':917,'historical_DeepSeek_pooled':False,
      'timing_scope':'precomputed cases; request + validation/rendering; excludes forecast/TreeSHAP recomputation and cold startup',
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    print(pd.DataFrame(summaries).to_string(index=False));print(pd.DataFrame(contrasts).to_string(index=False))

if __name__=='__main__':main()
