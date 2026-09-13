"""Frozen mutation families and explicit upstream-error boundary tests.

The gold labels are constructed from the mutation operation, not returned by the
detector. Development and challenge families are reported separately. This synthetic
contract audit is not a clinician-rated safety dataset or an adversarial proof.
"""
import copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.revision.grounding import validate,expected_output,study_score
from src.revision.data import sha256


def cases():
    for i in range(30):
        g=50.+i*8;rate=round((i%7-3)*.37,2)
        yield {'case_id':f'synthetic_{i}','values':{
            'observations':{'glucose_mgdl':g,'hr_bpm':65.+i,'input_lag_minutes':15 if i%2 else 0},
            'forecasts_mgdl':{'h30':g+7,'h60':g-12,'h90':g+(10 if i%2 else -10)},
            'rate_mgdl_per_min':rate,'study_score':study_score(g,rate),
            'shap_h30':[{'feature':f'glucose_mgdl_lag_{j*15}min','contribution_mgdl':float(j-2)} for j in range(3)]}}


def variants(case):
    base=expected_output(case)
    yield 'control','valid',json.dumps(base),False,case
    yield 'control','valid_scientific_notation',json.dumps(base).replace('50.0','5e1'),False,case
    changes=[('development','wrong_glucose',lambda d:d['observations'].update(glucose_mgdl=999.)),
             ('development','unit_field_swap',lambda d:d['observations'].update(glucose_mgdl=d['observations']['hr_bpm'])),
             ('development','horizon_swap',lambda d:d['forecasts_mgdl'].update(h30=d['forecasts_mgdl']['h60'])),
             ('development','wrong_tier',lambda d:d.update(study_tier=0)),
             ('development','missing_forecast',lambda d:d['forecasts_mgdl'].pop('h90')),
             ('challenge','shap_sign',lambda d:d['shap_h30'][0].update(contribution_mgdl=2.)),
             ('challenge','rate_interval_confusion',lambda d:d.update(rate_mgdl_per_min=d['rate_mgdl_per_min']*15+1)),
             ('challenge','numeric_string',lambda d:d['observations'].update(glucose_mgdl=str(d['observations']['glucose_mgdl']))),
             ('challenge','invented_interval',lambda d:d.update(ci95=[100,120])),
             ('challenge','causal_or_dose_advice',lambda d:d.update(advice='Heart rate caused this. Take 10 units of insulin.')),
             ('challenge','unicode_key',lambda d:d.update({'study_tіer':d.pop('study_tier')})),
             ('challenge','boolean_tier',lambda d:d.update(study_tier=True))]
    for split,name,change in changes:
        d=copy.deepcopy(base);change(d);yield split,name,json.dumps(d),True,case
    yield 'challenge','duplicate_key','{"study_tier":0,'+json.dumps(base)[1:],True,case
    yield 'challenge','empty_claims','{}',True,case
    yield 'challenge','nonfinite','{"study_score":NaN,'+json.dumps(base)[1:],True,case
    # The contract checker cannot detect an inaccurate source datum when it is faithfully copied.
    poisoned=copy.deepcopy(case);poisoned['values']['observations']['hr_bpm']+=40
    yield 'boundary','source_corruption',json.dumps(expected_output(poisoned)),True,poisoned


if __name__=='__main__':
    out=ROOT/'results/revision_v2/llm';out.mkdir(parents=True,exist_ok=True);rows=[];archive=[]
    for case in cases():
        for split,family,text,bad,input_case in variants(case):
            result=validate(text,input_case)
            rows.append({'case_id':case['case_id'],'split':split,'family':family,'gold_invalid':bad,
                         'rejected':not result['accepted'],'false_negative':bool(bad and result['accepted']),
                         'false_positive':bool(not bad and not result['accepted'])})
            archive.append({'case':input_case,'text':text,'split':split,'family':family,'gold_invalid':bad,'result':result})
    pd.DataFrame(rows).to_csv(out/'guardrail_challenge.csv',index=False)
    (out/'guardrail_challenge.json').write_text(json.dumps(archive,indent=2))
    manifest={'detector_sha256':sha256(ROOT/'src/revision/grounding.py'),'challenge_sha256':sha256(Path(__file__)),
              'note':'Frozen before final LLM generation. Boundary cases deliberately violate source truth, outside the typed contract.'}
    (out/'guardrail_freeze.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(rows).groupby(['split','family']).agg(n=('rejected','size'),rejected=('rejected','sum'),FN=('false_negative','sum'),FP=('false_positive','sum')).to_string())
