"""Descriptive heterogeneity and matched secondary comparisons, without causal labels."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.evaluation.metrics import summarize


def main():
    out=ROOT/'results/revision_v2';tables=out/'tables';tables.mkdir(exist_ok=True)
    per=pd.read_csv(tables/'per_patient_metrics.csv',dtype={'PtID':str,'seed':str})
    primary=pd.read_parquet(out/'all_predictions.parquet')
    profiles=[];correlations=[]
    for dataset in ('ddatshr','diadata'):
        root=ROOT/'data/processed/revision_v2/history_120'
        m=pd.read_parquet(root/f'{dataset}_samples.parquet');m.PtID=m.PtID.astype(str)
        x=np.load(root/f'{dataset}.npz')['X']
        m['abs_recent_rate']=np.abs(x[:,-1,0]-x[:,-2,0])/15
        p=m.groupby('PtID').agg(mean_glucose=('glucose_mgdl','mean'),sd_glucose=('glucose_mgdl','std'),
                               mean_hr=('hr_bpm','mean'),mean_abs_rate=('abs_recent_rate','mean'),
                               low_fraction=('glucose_mgdl',lambda x:(x<70).mean()),
                               high_fraction=('glucose_mgdl',lambda x:(x>180).mean())).reset_index()
        r=pd.read_csv(root/f'{dataset}_retention.csv',dtype={'PtID':str})
        p=p.merge(r,on='PtID');p['glucose_missing_fraction']=1-p.observed_glucose/p.grid_rows
        p['hr_missing_fraction']=1-p.observed_hr/p.grid_rows;p['dataset']=dataset
        if dataset=='ddatshr':
            demo=pd.read_csv(ROOT/'data/raw/ddatshr/population.csv',encoding='utf-8-sig')
            demo=demo.rename(columns={'Subject code number':'PtID','Age [yr]':'age','Gender [M=male F=female]':'sex',
                                      'HbA1C_1 [mmol/mol]':'first_reported_HbA1c_mmolmol'})
            demo.PtID=demo.PtID.astype(str)
            p=p.merge(demo[['PtID','age','sex','first_reported_HbA1c_mmolmol']],on='PtID',how='left')
        metric=per[(per.dataset==dataset)&(per.model=='RandomForest')&(per.condition=='cgm_hr')&(per.seed=='mean')&(per.horizon==30)]
        p=p.merge(metric[['PtID','MAE','RMSE']],on='PtID');profiles.append(p)
        for feature in ['eligible_samples','span_days','mean_glucose','sd_glucose','mean_abs_rate',
                        'glucose_missing_fraction','hr_missing_fraction','first_reported_HbA1c_mmolmol','age']:
            if feature in p:
                valid=p[[feature,'RMSE']].dropna()
                if len(valid)>3 and valid[feature].nunique()>1:
                    rho,pvalue=spearmanr(valid[feature],valid.RMSE)
                    correlations.append({'dataset':dataset,'feature':feature,'n_patients':len(valid),
                                         'rho':rho,'p_exploratory_unadjusted':pvalue})
    pd.concat(profiles,ignore_index=True).to_csv(tables/'patient_profiles.csv',index=False)
    pd.DataFrame(correlations).to_csv(tables/'heterogeneity_correlations_exploratory.csv',index=False)
    # A longer history changes eligibility and potentially the inner-validation subject.
    # Restrict sample IDs, and record whether validation allocation was identical.
    history=[]
    for path in (out/'sensitivity/history').glob('fits/*/*/*/seed_42/*/test_predictions.parquet'):
        long=pd.read_parquet(path);dataset=path.parts[-6];model=path.parts[-4];pt=path.parent.name
        short_path=out/'sensitivity/history_short_matched/fits'/dataset/'cgm_hr'/model/'seed_42'/pt/'test_predictions.parquet'
        short=pd.read_parquet(short_path)
        merged=long.merge(short,on='sample_id',suffixes=('_long','_short'))
        if len(merged)!=len(long):raise RuntimeError('4-hour cases must be a subset of 2-hour cases')
        for h in (30,60,90):
            np.testing.assert_array_equal(merged[f'true_{h}_long'],merged[f'true_{h}_short'])
            for label in ('long','short'):
                history.append({'dataset':dataset,'model':model,'PtID':pt,'history':240 if label=='long' else 120,
                                'horizon':h,**summarize(merged[f'true_{h}_long'],merged[f'pred_{h}_{label}'])})
    pd.DataFrame(history).to_csv(tables/'history_sensitivity_matched.csv',index=False)
    # Re-score source-held-out models after averaging seeds, at original patient level.
    transfer=[]
    paths=out/'sensitivity/source/fits/diadata/cgm_hr'
    for model in ('RandomForest','XGBoost'):
        for source in ('D1NAMO','HUPA-UCM'):
            fs=[pd.read_parquet(paths/model/f'seed_{s}'/f'SOURCE_{source}'/'test_predictions.parquet').sort_values('sample_id') for s in (42,1729,2026)]
            f=fs[0].copy();f.PtID=f.original_PtID
            for h in (30,60,90):
                f[f'pred_{h}']=np.mean([v[f'pred_{h}'].to_numpy() for v in fs],axis=0)
                for pt,d in f.groupby('PtID'):
                    transfer.append({'model':model,'held_out_source':source,'PtID':pt,'horizon':h,**summarize(d[f'true_{h}'],d[f'pred_{h}'])})
    pd.DataFrame(transfer).to_csv(tables/'source_transfer_seed_mean.csv',index=False)

if __name__=='__main__':main()
