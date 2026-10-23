"""Post-review reanalysis of saved participant summaries, never a raw rerun.

Writes only to a new destination. Retains input hashes and distinguishes pooled
counts from equal-participant estimates. Neural comparisons are exploratory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]

def holm(p):
    p = np.asarray(p); order = np.argsort(p); result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p)-np.arange(len(p)))))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {n: ROOT/'results/revision_v2/tables'/n for n in
              ['per_patient_metrics.csv', 'target_classification.csv', 'primary_comparisons.csv']}
    per = pd.read_csv(inputs['per_patient_metrics.csv'], dtype={'seed': str, 'PtID': str})
    per = per[per.seed.eq('mean')]
    rows = []
    for (cohort, model, horizon), group in per[per.model.isin(
            ['TFTEncoder', 'BiLSTMTransformer', 'CNNGRU'])].groupby(['dataset', 'model', 'horizon']):
        paired = group.pivot(index='PtID', columns='condition', values='MAE')
        assert paired[['cgm', 'cgm_hr']].notna().all().all()
        counts = group.pivot(index='PtID', columns='condition', values='n')
        assert (counts.cgm == counts.cgm_hr).all()
        delta = (paired.cgm_hr-paired.cgm).to_numpy()
        rng = np.random.default_rng(917)
        boot = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
        rows.append(dict(dataset=cohort, model=model, horizon=int(horizon), n_patients=len(delta),
                         cgm_MAE=paired.cgm.mean(), cgm_hr_MAE=paired.cgm_hr.mean(),
                         difference_MAE=delta.mean(), ci95_low=np.quantile(boot,.025),
                         ci95_high=np.quantile(boot,.975), patients_improved=int((delta<0).sum()),
                         p_raw=float(wilcoxon(delta,zero_method='pratt',method='auto').pvalue)))
    neural = pd.DataFrame(rows)
    neural['p_holm_exploratory_18'] = holm(neural.p_raw)
    neural.to_csv(args.output/'neural_hr_exploratory.csv', index=False)
    cls = pd.read_csv(inputs['target_classification.csv'], dtype={'seed': str, 'PtID': str})
    cls = cls[cls.seed.eq('mean') & cls.condition.eq('cgm_hr') & cls.model.eq('RandomForest') & cls.event.eq('low')]
    clinical = []
    for (cohort,horizon), group in cls.groupby(['dataset','horizon']):
        tp,fp,fn = (int(group[c].sum()) for c in ['tp','fp','fn'])
        clinical.append(dict(dataset=cohort,horizon=int(horizon),n_patients=len(group),tp=tp,fp=fp,fn=fn,
                             observed_positive=tp+fn,predicted_positive=tp+fp,
                             sensitivity_pooled=tp/(tp+fn) if tp+fn else None,
                             precision_pooled=tp/(tp+fp) if tp+fp else None,
                             sensitivity_macro=group.sensitivity.mean(),precision_macro=group.precision.mean(),
                             n_defined_sensitivity=int(group.sensitivity.notna().sum()),
                             n_defined_precision=int(group.precision.notna().sum())))
    pd.DataFrame(clinical).to_csv(args.output/'rf_hypoglycemia_saved_counts.csv',index=False)
    coverage=[]
    for (cohort,horizon), group in per[per.condition.eq('cgm_hr') & per.model.eq('RandomForest')].groupby(['dataset','horizon']):
        assert group.interval_coverage.notna().all()
        recovered=group.interval_coverage*group.n
        # Recover counts only if saved fractions are integer-consistent. This is
        # a consistency audit, not verification against individual predictions.
        assert np.allclose(recovered,np.rint(recovered),rtol=0,atol=1e-7)
        coverage.append(dict(dataset=cohort,horizon=int(horizon),n_patients=len(group),
                             evaluated_origins=int(group.n.sum()),
                             covered_origins_recovered=int(np.rint(recovered).sum()),
                             coverage_macro=group.interval_coverage.mean(),
                             coverage_pooled_recovered=np.rint(recovered).sum()/group.n.sum(),
                             verification='integer-consistent saved fractions; raw predictions absent'))
    pd.DataFrame(coverage).to_csv(args.output/'coverage_saved_fraction_audit.csv',index=False)
    provenance={'analysis':'post-review reaggregation of historical saved participant tables',
                'not_performed':['new model fitting','raw prediction verification','Qwen inference','human evaluation'],
                'bootstrap_seed':917,'bootstrap_replicates':10000,
                'neural_family':'18 exploratory comparisons; separate from primary family; TFT choice is post hoc',
                'inputs':{n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in inputs.items()},
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(neural[neural.model.eq('TFTEncoder')].to_string(index=False))
    print(pd.DataFrame(clinical).to_string(index=False))
    print(pd.DataFrame(coverage).to_string(index=False))

if __name__ == '__main__':
    main()
