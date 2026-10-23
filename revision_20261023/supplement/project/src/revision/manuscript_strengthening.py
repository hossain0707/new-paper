"""Resolve additional manuscript numbers exclusively from completed evidence."""
import json
import pandas as pd

def values(root,table):
 out=root/'results/revision_v3';m=pd.read_csv(out/'tables/subject_macro_metrics.csv',dtype={'seed':str});m=m[m.seed=='mean']
 c=pd.read_csv(out/'tables/descriptive_comparisons.csv');a=json.loads((out/'external/data_audit.json').read_text());freeze=json.loads((out/'chronos/freeze.json').read_text())
 v={'azt_n':a['patients'],'azt_samples':f"{a['origins']:,}",'chronos_package':freeze['package'],'chronos_revision':freeze['revision']}
 rows=[]
 for name in ['Persistence','LinearTrend','Ridge','RandomForest','XGBoost','TFTEncoder','BiLSTMTransformer','CNNGRU','Stack']:
  d=m[(m.dataset=='azt1d')&(m.model==name)].sort_values('horizon');assert len(d)==3
  rows.append([name,*[f'{r.MAE:.2f} [{r.MAE_ci_low:.2f}, {r.MAE_ci_high:.2f}]' for r in d.itertuples()]])
 v['table_external']=table(['Model','30 min','60 min','90 min'],rows)
 best=[]
 for h,d in m[m.dataset=='azt1d'].groupby('horizon'):
  r=d.loc[d.MAE.idxmin()];best.append(f'{h} minutes: {r.model}, {r.MAE:.2f} mg/dL')
 text='AZT1D’s lowest point-estimate MAEs were '+ '; '.join(best)+'. '
 for dataset in ['ddatshr','diadata']:
  d=m[(m.dataset==dataset)&(m.model=='Chronos2')].sort_values('horizon');text+=f'Chronos-2 MAE on {dataset} was '+ '/'.join(f'{x:.2f}' for x in d.MAE)+' mg/dL. '
  d=c[(c.dataset==dataset)&(c.comparison=='HR_added_three_validation')].sort_values('horizon');text+=f'Three-validator RF HR-minus-CGM differences on {dataset} were '+ '/'.join(f'{x:+.2f}' for x in d.difference_MAE)+' mg/dL. '
 text+='These point estimates do not establish universal architecture superiority or HR equivalence. The supplementary tables retain all seeds, participant errors, glycemic ranges, Clarke zones and paired descriptive intervals.'
 v['additional_results_text']=text
 latency=pd.read_csv(out/'latency/summary.csv');d=latency[latency.component=='total_s'];assert len(d)==3 and d.n.eq(30).all()
 v['table_latency']=table(['Arm','Accepted','Median (s)','95th percentile (s)','Maximum (s)','Deadline misses'],[[r.arm,f'{r.accepted}/{r.n}',f'{r.median:.3f}',f'{r.p95:.3f}',f'{r.maximum:.3f}',r.deadline_missed] for r in d.itertuples()])
 measurements=pd.read_csv(out/'latency/measurements.csv')
 people=len(measurements[['dataset','PtID']].drop_duplicates())
 v['latency_results_text']=f'The 30 cases represented {people} participants. Median total processing times were '+ '; '.join(f'{r.arm}: {r.median:.3f} s' for r in d.itertuples())+f'. There were {int(d.deadline_missed.sum())} observed deadline misses and {int(d.request_errors.sum())} request errors across 90 measured runs. This small warm replay cannot establish hard real-time guarantees or patient benefit.'
 (out/'manuscript_values.json').write_text(json.dumps(v,indent=2));return v
