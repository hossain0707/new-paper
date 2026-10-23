"""Regenerate current Figure 2 solely from Qwen raw audit and synthetic contracts."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(sys.argv[1]);OUT=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.revision.grounding import validate
raw=ROOT/'results/revision_v2/llm/guardrail_challenge.json';items=json.loads(raw.read_text())
rows=[]
for r in items:
 check=validate(r['text'],r['case'])
 rows.append({'split':r['split'],'family':r['family'],'rejected':int(not check['accepted'])})
g=pd.DataFrame(rows).groupby('split').rejected.agg(['sum','size']);g.to_csv(OUT/'synthetic_revalidation.csv')
q=pd.read_csv(ROOT/'results/revision_v4/spark_20261022/qwen/test_summary/arm_summary.csv')
order=['deterministic','zero_calculated_plain','zero_calculated_schema','zero_supplied_plain','zero_supplied_schema','few_calculated_plain','few_calculated_schema','few_supplied_plain','few_supplied_schema'];q=q.set_index('arm').loc[order]
labels=['Deterministic','Zero / calc / plain','Zero / calc / schema','Zero / supplied / plain','Zero / supplied / schema','Few / calc / plain','Few / calc / schema','Few / supplied / plain','Few / supplied / schema']
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(1,2,figsize=(11.5,4.5),gridspec_kw={'width_ratios':[1.6,1]})
ax[0].barh(labels,100*q.acceptance_case,color=['#737373']+['#39789a']*4+['#9d633a']*4);ax[0].invert_yaxis();ax[0].set_xlim(0,120);ax[0].set_xlabel('Complete-contract acceptance (%)');ax[0].set_title('A  Qwen test outputs')
for i,r in enumerate(q.itertuples()):ax[0].text(100*r.acceptance_case+1,i,f'{r.accepted}/111',va='center',fontsize=8)
splits=['control','development','challenge','boundary'];labels2=['Valid controls','Development violations','Challenge violations','Corrupted source']
counts=g.loc[splits];perc=100*counts['sum']/counts['size'];ax[1].barh(labels2,perc,color='#737373');ax[1].invert_yaxis();ax[1].set_xlim(0,125);ax[1].set_xlabel('Rejected (%)');ax[1].set_title('B  Synthetic checker audit')
for i,(a,b) in enumerate(zip(counts['sum'],counts['size'])):ax[1].text(max(1,100*a/b+1),i,f'{a}/{b} ({100*a/b:.0f}%)',va='center',fontsize=8)
for a in ax:a.spines[['top','right']].set_visible(False)
fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'figure2_qwen.{ext}',dpi=200,bbox_inches='tight')
(OUT/'figure2_provenance.json').write_text(json.dumps({'synthetic_archive_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'validator_sha256':hashlib.sha256((ROOT/'src/revision/grounding.py').read_bytes()).hexdigest(),'synthetic_replayed':len(rows),'Qwen_only':True},indent=2))
print(g)
