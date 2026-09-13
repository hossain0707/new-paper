"""Resolve manuscript tables and numerical claims from the completed evidence tables."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(c).replace('|',' / ') for c in r)+' |' for r in rows])


def main():
    out=ROOT/'results/revision_v2';tab=out/'tables';values={}
    macro=pd.read_csv(tab/'subject_macro_metrics.csv',dtype={'seed':str})
    per=pd.read_csv(tab/'per_patient_metrics.csv',dtype={'seed':str,'PtID':str})
    pooled=pd.read_csv(tab/'pooled_metrics.csv');comp=pd.read_csv(tab/'primary_comparisons.csv')
    profiles=pd.read_csv(tab/'patient_profiles.csv',dtype={'PtID':str})
    gen=pd.read_csv(out/'llm/generation_metrics.csv',dtype={'PtID':str})
    audit=pd.read_csv(out/'llm/guardrail_challenge.csv');test=gen[gen.split=='test']
    main=macro[(macro.seed=='mean')&(macro.condition=='cgm_hr')]
    vector=lambda d,col:'/'.join(f'{v:.2f}' for v in d.sort_values('horizon')[col])
    retain=[]
    for dataset,short in [('ddatshr','dd'),('diadata','dia')]:
        a=json.loads((ROOT/f'data/processed/revision_v2/history_120/{dataset}_audit.json').read_text())
        values[short+'_n']=a['patients'];values[short+'_samples']=f"{a['samples']:,}"
        values[short+'_rf_mae']=vector(main[(main.dataset==dataset)&(main.model=='RandomForest')],'MAE')
        ret=pd.read_csv(ROOT/f'data/processed/revision_v2/history_120/{dataset}_retention.csv')
        for source,d in ret.groupby('source'):
            eligible=d[d.retained];retain.append([dataset,source,len(d),len(eligible),f'{eligible.eligible_samples.sum():,}'])
    external_audit=json.loads((ROOT/'results/revision_v3/external/data_audit.json').read_text())
    retain.append(['azt1d (additional)','AZT1D',25,external_audit['patients'],f"{external_audit['origins']:,}"])
    values['table_retention']=table(['Cohort','Source','Source participants','Retained','Eligible origins'],retain)
    values['retention_text']=(f"The corrected primary analysis retained {values['dd_n']} DDATSHR participants "
                              f"({values['dd_samples']} origins) and {values['dia_n']} DiaData participants "
                              f"({values['dia_samples']} origins). The Medtronic parser quarantined 1,540 malformed records. "
                              "DiaData excluded all 18 DDATSHR-source participants before eligibility assessment. "
                              "Excluded participants and per-source availability are retained in the supplementary tables.")
    rows=[]
    for dataset in ['ddatshr','diadata']:
        for model in ['Persistence','LinearTrend','Ridge','RandomForest','XGBoost','TFTEncoder','BiLSTMTransformer','CNNGRU','Stack','TreeStack']:
            d=main[(main.dataset==dataset)&(main.model==model)].sort_values('horizon')
            assert len(d)==3
            rows.append([dataset,model,*[f'{v:.2f}' for v in d.MAE]])
    values['table_forecasts']=table(['Cohort','Model','30 min','60 min','90 min'],rows)
    best=[]
    for (dataset,h),d in main.groupby(['dataset','horizon']):
        b=d.loc[d.MAE.idxmin()];best.append(f'{dataset} at {h} minutes: {b.model} ({b.MAE:.2f} mg/dL)')
    values['forecast_results_text']='The lowest point-estimate MAEs were '+ '; '.join(best)+'. These rankings are descriptive, not multiplicity-adjusted pairwise superiority claims.'
    labels={'HR_added':'RF HR − RF CGM','Stack_vs_RF':'Stack − RF','RF_vs_persistence':'RF − persistence'}
    rows=[]
    for r in comp.itertuples():
        rows.append([r.dataset,r.horizon,labels[r.comparison],f'{r.difference_MAE:+.2f}',f'[{r.ci95_low:+.2f}, {r.ci95_high:+.2f}]',f'{r.p_holm:.4g}'])
    values['table_comparisons']=table(['Cohort','Horizon','Contrast','MAE difference','95% interval','Holm p'],rows)
    sentences=[]
    for name in labels:
        d=comp[comp.comparison==name];better=d[(d.p_holm<.05)&(d.difference_MAE<0)];worse=d[(d.p_holm<.05)&(d.difference_MAE>0)]
        sentences.append(f"{labels[name]} showed a Holm-significant improvement in {len(better)}/{len(d)} cohort–horizon comparisons "
                         f"and a Holm-significant deterioration in {len(worse)}/{len(d)}")
    values['hypothesis_results_text']='; '.join(sentences)+'. Failure to reject is not evidence of equivalence.'
    hr=comp[comp.comparison=='HR_added'];st=comp[comp.comparison=='Stack_vs_RF']
    values['primary_summary']=(f"After correction across 18 tests, adding HR improved RF in {int(((hr.p_holm<.05)&(hr.difference_MAE<0)).sum())}/6 "
                               f"cohort–horizon comparisons, while stacking improved RF in {int(((st.p_holm<.05)&(st.difference_MAE<0)).sum())}/6.")
    stack_locations=[]
    for dataset,d in st.groupby('dataset'):
        count=int(((d.p_holm<.05)&(d.difference_MAE<0)).sum())
        stack_locations.append(f'{dataset}: {count}/{len(d)} horizons')
    values['primary_summary']+=' The stack findings were '+ '; '.join(stack_locations)+'.'
    seeds=macro[(macro.seed!='mean')&(macro.condition=='cgm_hr')&(macro.model=='RandomForest')&(macro.horizon==30)]
    values['seed_results_text']='Across individual RF training seeds, 30-minute subject-macro MAE ranged '+ '; '.join(
        f'{d.MAE.min():.2f}–{d.MAE.max():.2f} mg/dL for {dataset}' for dataset,d in seeds.groupby('dataset'))+'. Full seed-specific results are supplied, rather than treating seeds as additional patients.'
    r=profiles[(profiles.dataset=='ddatshr')&(profiles.PtID=='907')].iloc[0]
    d=profiles[profiles.dataset=='ddatshr'];rank=int(d.RMSE.rank(ascending=False,method='min')[d.PtID.eq('907')].iloc[0])
    values['heterogeneity_text']=(f"DDATSHR RF 30-minute participant RMSE ranged from {d.RMSE.min():.2f} to {d.RMSE.max():.2f} mg/dL. "
        f"Participant 907 had RMSE {r.RMSE:.2f} mg/dL (rank {rank} of {len(d)}, highest error first), "
        f"{int(r.eligible_samples):,} eligible origins and {r.span_days:.1f} recorded days. "
        f"Its source-grid glucose and HR missing fractions were {100*r.glucose_missing_fraction:.1f}% and {100*r.hr_missing_fraction:.1f}%, "
        f"and eligible-input glucose SD was {r.sd_glucose:.2f} mg/dL. Actual sensor labels were {r.sensors.replace('|',' and ')}. "
        "These measurements replace the earlier inference from a small, device-selected sample; they do not identify a physiological cause.")
    correlations=pd.read_csv(tab/'heterogeneity_correlations_exploratory.csv')
    details=[]
    for dataset,d in correlations.groupby('dataset'):
        indexed=d.set_index('feature')
        details.append(f'{dataset}: recent absolute glucose rate rho={indexed.loc["mean_abs_rate","rho"]:.2f}, glucose SD rho={indexed.loc["sd_glucose","rho"]:.2f}, glucose missing fraction rho={indexed.loc["glucose_missing_fraction","rho"]:.2f}, HR missing fraction rho={indexed.loc["hr_missing_fraction","rho"]:.2f}')
    values['heterogeneity_text']+=' Exploratory Spearman associations with participant RMSE were '+ '; '.join(details)+'. These cross-participant associations are unadjusted, may reflect source composition and do not establish causes of an individual participant’s error.'
    rows=[]
    for r in pooled[(pooled.model=='RandomForest')&(pooled.condition=='cgm_hr')].itertuples():
        rows.append([r.dataset,r.horizon,f'{r.RMSE:.2f}',' / '.join(f'{getattr(r,"Clarke_"+z+"_pct"):.2f}' for z in 'ABCDE')])
    values['table_clarke']=table(['Cohort','Horizon','Pooled RMSE','Clarke A / B / C / D / E (%)'],rows)
    values['clarke_text']='Zone A alone is not a complete error characterization: potentially important C, D and E errors remain visible in the full table and are not absorbed into a single A+B percentage.'
    history=pd.read_csv(tab/'history_sensitivity_matched.csv');transfer=pd.read_csv(tab/'source_transfer_seed_mean.csv')
    rows=[]
    for (dataset,model,hist),d in history.groupby(['dataset','model','history']):
        m=d.groupby('horizon').MAE.mean();rows.append([f'{dataset}, {hist} min history',model,len(d.PtID.unique()),'/'.join(f'{x:.2f}' for x in m)])
    for (source,model),d in transfer.groupby(['held_out_source','model']):
        m=d.groupby('horizon').MAE.mean();rows.append([f'Held-out source: {source}',model,len(d.PtID.unique()),'/'.join(f'{x:.2f}' for x in m)])
    values['table_sensitivity']=table(['Exploratory condition','Model','Patients','Macro MAE 30/60/90'],rows)
    values['sensitivity_text']='The four-hour requirement retained 17 DDATSHR and 26 DiaData participants. The two history versions use the same longer-history eligible origins and participant allocations. Source transfer tests both directions between the seven primary D1NAMO and 22 HUPA-UCM participants; its training-source sizes are consequently unequal.'
    intervals=main[main.model=='RandomForest']
    values['interval_text']='Descriptive RF validation-residual interval coverage was '+ '; '.join(
        f'{dataset}: '+ '/'.join(f'{100*x:.1f}%' for x in d.sort_values('horizon').interval_coverage)+
        ' at 30/60/90 minutes (mean widths '+ '/'.join(f'{x:.1f}' for x in d.sort_values('horizon').interval_width)+' mg/dL)'
        for dataset,d in intervals.groupby('dataset'))+'. These observed coverages are not assumed to equal their nominal 95% level.'
    arms={'deterministic':'template_accept','zero_shot':'zero_accept','few_shot':'few_accept'};rows=[]
    for arm,key in arms.items():
        d=test[test.arm==arm];accepted=int(d.accepted.sum());n=len(d);assert n>0
        values[key]=f'{accepted}/{n}'
        rows.append([arm,n,accepted,f'{100*accepted/n:.1f}%',int(d.error.notna().sum()),f'{d.latency_s.median():.3f}',f'{d.latency_s.quantile(.95):.3f}'])
    values['test_cases']=test.case_id.nunique()
    values['table_generation']=table(['Arm','Cases','Accepted','Acceptance','Request errors','Median s','p95 s'],rows)
    dev=gen[gen.split=='development'];selection=pd.read_csv(out/'llm/case_selection.csv')
    assert len(test)==3*values['test_cases'] and values['test_cases']==len(selection[selection.split=='test'])
    values['generation_text']=(f"The fixed selection contained {len(selection)} cases from {selection.PtID.nunique()} people; "
        f"{values['test_cases']} cases from {test.PtID.nunique()} people formed the final test set. "
        f"Development used {len(selection[selection.split=='development'])} cases from {selection[selection.split=='development'].PtID.nunique()} other people. "
        f"Zero-shot and few-shot complete-contract acceptance was {values['zero_accept']} and {values['few_accept']}, respectively. "
        "The deterministic arm supplies a direct comparator with no language-model inference. All raw responses and request failures are retained, including rejected outputs.")
    intervals_gen=pd.read_csv(out/'llm/acceptance_cluster_intervals.csv')
    descriptions=[]
    for arm in ('zero_shot','few_shot'):
        r=intervals_gen[(intervals_gen.split=='test')&(intervals_gen.arm==arm)].iloc[0]
        descriptions.append(f'{arm}: {100*r.macro_acceptance:.1f}% (95% interval {100*r.ci95_low:.1f}–{100*r.ci95_high:.1f}%)')
    values['generation_uncertainty_text']='Giving each test participant equal weight, contract acceptance was '+ '; '.join(descriptions)+'. Intervals use 10,000 participant bootstrap resamples and are descriptive; three state-selected cases per person are not independent patients.'
    errors=pd.read_csv(out/'llm/rejection_analysis.csv')
    errors=errors[(errors.split=='test')&(errors.arm!='deterministic')]
    counts=errors.drop_duplicates(['case_id','arm','category']).groupby('category').size()
    values['generation_errors_text']=('Rejected final completions included '+ '; '.join(f'{category} mismatches in {count} completions' for category,count in counts.items())+'. Categories can overlap within a completion. The prompt and detector were not modified in response to these final-test failures.') if len(counts) else 'No contract rejection was observed on these final cases; this does not exclude untested failures.'
    c=audit[audit.split=='challenge'];b=audit[audit.split=='boundary'];control=audit[audit.split=='control']
    values.update(challenge_rejected=int(c.rejected.sum()),challenge_total=len(c),boundary_accepted=int((~b.rejected).sum()),boundary_total=len(b))
    values['challenge_text']=(f"The separately designated challenge families produced {len(c)} invalid contracts; {int(c.rejected.sum())} were rejected. "
        f"Among {len(control)} valid controls, {int(control.false_positive.sum())} were rejected. "
        f"All {len(b)} source-corruption boundary cases were accepted despite deliberately wrong underlying HR values. "
        "Thus zero observed within-contract false negatives would not imply zero end-to-end factual errors. The development-family results are reported separately in the supplement.")
    values['discussion_results']=values['primary_summary']+' '+values['hypothesis_results_text']
    values['conclusion_text']=values['primary_summary']+' '+f"Typed generation accepted {values['zero_accept']} zero-shot and {values['few_accept']} few-shot test completions, while explicit source-corruption tests remained undetectable to a source-relative checker."
    from src.revision.manuscript_strengthening import values as strengthening_values
    values.update(strengthening_values(ROOT,table))
    template=(ROOT/'manuscript/revision_template.md').read_text()
    keys=set(re.findall(r'{{(\w+)}}',template));assert keys<=set(values),keys-set(values)
    for key in keys:template=template.replace('{{'+key+'}}',str(values[key]))
    assert '{{' not in template
    # Reorder references by first citation after restoring the original section sequence.
    body,bibliography=template.split('## References\n',1)
    refs={int(m.group(1)):m.group(2) for m in re.finditer(r'^(\d+)\. (.+)$',bibliography,re.M)}
    order=[]
    def cite(match):
        old=[]
        for group in match.group(1).split(','):
            group=group.strip()
            if '–' in group or '-' in group:
                lo,hi=map(int,re.split('[–-]',group));old.extend(range(lo,hi+1))
            else:old.append(int(group))
        for number in old:
            assert number in refs
            if number not in order:order.append(number)
        return '['+', '.join(str(order.index(number)+1) for number in old)+']'
    body=re.sub(r'\[(\d+(?:\s*[,–-]\s*\d+)*)\]',cite,body)
    assert set(order)==set(refs),set(refs)-set(order)
    template=body+'## References\n\n'+'\n'.join(f'{i}. {refs[n]}' for i,n in enumerate(order,1))+'\n'
    (ROOT/'manuscript/manuscript_REVISED_EVIDENCE.md').write_text(template)
    (out/'manuscript_values.json').write_text(json.dumps(values,indent=2))
    print('Wrote evidence-linked manuscript; human declarations remain explicitly pending.')

if __name__=='__main__':main()
