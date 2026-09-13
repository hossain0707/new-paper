"""Recompute presentation metadata from immutable generation outputs; no inference calls."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from src.revision.grounding import validate
from src.revision.data import sha256


if __name__=='__main__':
    out=ROOT/'results/revision_v2/llm';rows=[];changes=[];rejections=[]
    frozen=json.loads((out/'prompt_freeze.json').read_text())
    assert frozen['detector_sha256']==sha256(ROOT/'src/revision/grounding.py')
    for path in sorted((out/'responses').glob('*.json')):
        r=json.loads(path.read_text());case=json.loads((out/'cases'/f"{r['case_id']}.json").read_text())
        validation=validate(r['raw_text'],case)
        assert validation['accepted']==r['validation']['accepted']
        origin='deterministic_template' if r['arm']=='deterministic' else ('llm' if validation['accepted'] else 'template_fallback')
        if r['origin']!=origin:
            changes.append({'file':path.name,'old_origin':r['origin'],'new_origin':origin})
            r['origin_before_presentation_correction']=r['origin'];r['origin']=origin
            path.write_text(json.dumps(r,indent=2))
        rows.append({k:r[k] for k in ['case_id','dataset','PtID','split','state','arm','latency_s','error','origin']}|
                    {'accepted':validation['accepted'],'verified_leaves':validation['verified_leaves']})
        for error in validation['errors']:
            category=('direction' if error.startswith('.direction') else 'tier' if error.startswith('.study_tier')
                      else 'SHAP' if error.startswith('.shap_') else 'format' if error.startswith('parse:') else 'other_contract')
            rejections.append({'case_id':r['case_id'],'PtID':r['PtID'],'split':r['split'],'arm':r['arm'],'category':category,'error':error})
    pd.DataFrame(rows).to_csv(out/'generation_metrics.csv',index=False)
    correction=out/'origin_metadata_correction.json'
    if changes:correction.write_text(json.dumps({'note':'Deterministic baseline origin corrected; raw requests, responses, acceptance and latency unchanged.','changes':changes},indent=2))
    frame=pd.DataFrame(rows);pd.DataFrame(rejections).to_csv(out/'rejection_analysis.csv',index=False)
    patient=frame.groupby(['split','dataset','PtID','arm']).accepted.mean().reset_index()
    patient.to_csv(out/'per_patient_acceptance.csv',index=False)
    estimates=[];rng=np.random.default_rng(917)
    for (split,arm),d in patient.groupby(['split','arm']):
        a=d.accepted.to_numpy();means=a[rng.integers(len(a),size=(10000,len(a)))].mean(axis=1)
        estimates.append({'split':split,'arm':arm,'n_patients':len(a),'macro_acceptance':a.mean(),
                          'ci95_low':np.quantile(means,.025),'ci95_high':np.quantile(means,.975)})
    pd.DataFrame(estimates).to_csv(out/'acceptance_cluster_intervals.csv',index=False)
    print(frame.groupby(['split','arm']).agg(n=('accepted','size'),accepted=('accepted','sum'),median_s=('latency_s','median')).to_string())
