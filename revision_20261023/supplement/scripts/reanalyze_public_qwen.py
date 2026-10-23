"""Recompute published participant-bootstrap Qwen statistics from public audit rows."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(sys.argv[1]);OUT=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=False)
d=pd.read_csv(ROOT/'Qwen/test_request_audit.csv');d.accepted=d.accepted.astype(int)
def interval(x):
 a=np.asarray(x);r=np.random.default_rng(917);b=a[r.integers(0,len(a),size=(10000,len(a)))].mean(axis=1)
 return a.mean(),*np.quantile(b,[.025,.975])
rows=[]
for arm,g in d.groupby('arm'):
 mean,lo,hi=interval(g.groupby('participant_group').accepted.mean())
 rows.append(dict(arm=arm,n=len(g),accepted=int(g.accepted.sum()),macro=mean,low=lo,high=hi,median_s=g.request_s.median()))
pd.DataFrame(rows).to_csv(OUT/'arms.csv',index=False)
d=d[d.arm!='deterministic'].copy();d['few']=d.arm.str.startswith('few');d['supplied']=d.arm.str.contains('_supplied_');d['schema']=d.arm.str.endswith('_schema');rows=[]
for factor in ['few','supplied','schema']:
 p=d.groupby(['participant_group',factor]).accepted.mean().unstack();diff=p[True]-p[False];mean,lo,hi=interval(diff)
 rows.append(dict(factor=factor,effect=mean,low=lo,high=hi,p=wilcoxon(diff,zero_method='pratt',method='auto').pvalue))
pv=np.array([r['p'] for r in rows]);order=np.argsort(pv);adj=np.empty(3);adj[order]=np.minimum(1,np.maximum.accumulate(pv[order]*(3-np.arange(3))))
old=pd.read_csv(ROOT/'Qwen/test_factor_contrasts.csv').set_index('factor')
for r,ap in zip(rows,adj):
 r['holm']=ap
 for key,column in [('effect','acceptance_effect'),('low','ci95_low'),('high','ci95_high'),('holm','p_holm_exploratory_3')]:assert abs(r[key]-old.loc[r['factor'],column])<1e-10,(r,key)
pd.DataFrame(rows).to_csv(OUT/'factors.csv',index=False)
(OUT/'VERIFIED.json').write_text(json.dumps({'requests':len(d),'factor_statistics_match':True,'source':'public deidentified decision/timing audit; raw payload validation is a separate private-artifact check'},indent=2))
print('888 requests and factor estimates match public audit')
