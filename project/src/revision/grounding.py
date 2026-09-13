"""Fail-closed typed grounding and controlled rendering, not clinical adjudication.

Arbitrary model prose is never marked factual. Only the exact declared fields are
eligible for rendering. A rejected completion is replaced by the deterministic source
template, and is counted as an LLM rejection, never an LLM factual success.
"""
import copy
import json
import math


def study_score(glucose,rate):
    """Legacy-inspired synthetic consistency task; NOT a validated risk score."""
    base=85 if glucose<=70 or glucose>=250 else (55 if glucose<=80 or glucose>=200 else 20)
    return round(min(100,base+min(15,abs(rate)*5)),1)


def study_tier(score):
    if not math.isfinite(score) or not 0<=score<=100:raise ValueError('Score outside study domain')
    return 1+sum(score>=b for b in (40,60,80,95))


def expected_output(case):
    v=case['values'];delta=v['forecasts_mgdl']['h90']-v['observations']['glucose_mgdl']
    return {**copy.deepcopy(v),'study_tier':study_tier(v['study_score']),
            'direction':'rising' if delta>5 else ('falling' if delta < -5 else 'stable')}


def _unique(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('Duplicate key: '+k)
        d[k]=v
    return d


def validate(text,case):
    """No coercion, pooled-number matching, guideline bypass, or vacuous pass."""
    errors=[];checked=0;total=0
    target=expected_output(case)
    def compare(actual,expected,path=''):
        nonlocal checked,total
        if isinstance(expected,dict):
            if not isinstance(actual,dict):errors.append(path+': expected object');return
            for k in set(actual)-set(expected):errors.append(path+'.'+k+': undeclared field')
            for k,v in expected.items():
                if k not in actual:errors.append(path+'.'+k+': missing')
                else:compare(actual[k],v,path+'.'+k)
        elif isinstance(expected,list):
            if not isinstance(actual,list) or len(actual)!=len(expected):errors.append(path+': wrong array');return
            for i,(a,e) in enumerate(zip(actual,expected)):compare(a,e,path+f'[{i}]')
        else:
            total+=1
            if isinstance(expected,(int,float)):
                integer_field=path in ('.study_tier','.observations.input_lag_minutes')
                ok=(type(actual) is int if integer_field else type(actual) in (int,float)) and math.isfinite(actual) and abs(actual-expected)<.00001
            else:ok=type(actual) is type(expected) and actual==expected
            if ok:checked+=1
            else:errors.append(path+': value/type mismatch')
    try:
        parsed=json.loads(text,object_pairs_hook=_unique,parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
        compare(parsed,target)
    except (ValueError,TypeError,RecursionError) as e:
        parsed=None;errors.append('parse: '+str(e)[:200])
    return {'accepted':not errors,'errors':errors,'verified_leaves':checked,'compared_leaves':total,'parsed':parsed}


def render(verified):
    """Render ONLY a validated object or expected_output(case). No free prose path."""
    o=verified['observations'];p=verified['forecasts_mgdl']
    shap='; '.join(f"{s['feature']}: {s['contribution_mgdl']:+.1f} mg/dL" for s in verified['shap_h30'])
    return (f"Latest available glucose: {o['glucose_mgdl']:.1f} mg/dL; heart rate: {o['hr_bpm']:.1f} bpm. "
            f"Input delay: {o['input_lag_minutes']} minutes. RF forecasts at 30, 60 and 90 minutes: "
            f"{p['h30']:.1f}, {p['h60']:.1f} and {p['h90']:.1f} mg/dL. "
            f"Recent glucose change: {verified['rate_mgdl_per_min']:.2f} mg/dL/min. "
            f"Forecast direction relative to the latest input: {verified['direction']}. "
            f"Study-only score: {verified['study_score']:.1f}; study tier {verified['study_tier']}. "
            f"Largest listed SHAP contributions to the 30-minute RF glucose-change forecast: {shap}. "
            "SHAP describes model attribution, not physiological causation. "
            "This is a retrospective research output, not a treatment recommendation or dispatched alert.")


def deliver(text,case):
    check=validate(text,case)
    output=check['parsed'] if check['accepted'] else expected_output(case)
    return {'validation':check,'origin':'llm' if check['accepted'] else 'template_fallback','narrative':render(output)}


SYSTEM_PROMPT='''Return one JSON object only, without Markdown, commentary, new fields, advice, or invented intervals.
Copy every value in input.values exactly, preserving units in the field names, every forecast horizon,
and the three SHAP entries in their supplied order. Add exactly two fields:
study_tier: integer 1 for 0<=score<40, 2 for 40<=score<60, 3 for 60<=score<80,
4 for 80<=score<95, 5 for 95<=score<=100, using study_score.
direction: "rising" if h90 minus observations.glucose_mgdl >5, "falling" if <-5,
otherwise "stable". These rules are an artificial consistency task, not validated clinical triage.
The observations are the last available input; the forecasts are RF predictions, not measurements.
shap_h30 contributions explain the RF change prediction, not physiology. Do not interpret SHAP as causation.
The only output keys are observations, forecasts_mgdl, rate_mgdl_per_min, study_score,
shap_h30, study_tier, direction.'''
