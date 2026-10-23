"""Readable figures regenerated from historical aggregate evidence only."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v4/local_20261022/figures'
OUT.mkdir(parents=True,exist_ok=True)
V2=ROOT/'results/revision_v2'; V3=ROOT/'results/revision_v3'
LABEL={'RandomForest':'Random forest','LinearTrend':'Linear trend','TFTEncoder':'TFT-inspired encoder',
       'BiLSTMTransformer':'BiLSTM–Transformer','CNNGRU':'CNN–GRU','TreeStack':'Tree stack',
       'ddatshr':'DDATSHR','diadata':'DiaData','zero_shot':'Zero-shot','few_shot':'Few-shot','deterministic':'Deterministic'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
def save(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf','svg']: fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
boxes=[(.2,4.4,'Observed CGM and heart rate\nRecorded source data'),(5.3,4.4,'Availability-aware processing\nObserved targets on a 15-minute grid'),
       (.2,2.5,'Participant-held-out forecasts\nTraining-only scaling'),(5.3,2.5,'Participant-level evaluation\nMatched HR and history analyses'),
       (.2,.5,'Held-out RF and TreeSHAP\nParticipant-separated language cases'),(5.3,.5,'Optional local LLM\nTyped validator and controlled rendering\nDeterministic comparator and fallback')]
for x,y,t in boxes:
    ax.add_patch(FancyBboxPatch((x,y),4.2,1.15,boxstyle='round,pad=.08',facecolor='#edf3f8',edgecolor='#4477AA'))
    ax.text(x+2.1,y+.575,t,ha='center',va='center',fontsize=8.5)
for p,q in [((4.5,4.97),(5.15,4.97)),((7.4,4.25),(2.3,3.9)),((4.5,3.07),(5.15,3.07)),((2.3,2.35),(2.3,1.8)),((4.5,1.07),(5.15,1.07))]:
    ax.annotate('',q,p,arrowprops={'arrowstyle':'->','color':'#555','lw':1.2})
save(fig,'figure_1')
gen=pd.read_csv(V2/'llm/generation_metrics.csv');gen=gen[gen.split.eq('test')]
challenge=pd.read_csv(V2/'llm/guardrail_challenge.csv')
fig,axes=plt.subplots(1,2,figsize=(8,3.5))
arms=['deterministic','zero_shot','few_shot']
groups=[gen[gen.arm.eq(a)] for a in arms]
values=[100*g.accepted.mean() for g in groups]
axes[0].bar(range(3),values,color=['#666','#4477AA','#228833'])
axes[0].set(xticks=range(3),xticklabels=[LABEL[a] for a in arms],ylim=(0,119),ylabel='Complete-contract acceptance (%)')
for i,g in enumerate(groups):axes[0].text(i,values[i]+2,f'{int(g.accepted.sum())}/{len(g)}',ha='center')
names=[];rates=[];counts=[]
for split,g in challenge.groupby('split',sort=False):
    names.append({'control':'Valid\ncontrols','development':'Development\nviolations','challenge':'Challenge\nviolations','boundary':'Corrupted\nsource','source_corruption':'Corrupted\nsource'}.get(split,split.replace('_','\n')))
    rates.append(100*g.rejected.mean());counts.append(f'{int(g.rejected.sum())}/{len(g)}')
axes[1].bar(range(len(names)),rates,color='#AA6677')
axes[1].set(xticks=range(len(names)),xticklabels=names,ylim=(0,119),ylabel='Synthetic contract rejection (%)')
for i,(rate,count) in enumerate(zip(rates,counts)):axes[1].text(i,rate+2,f'{rate:.0f}%\n{count}',ha='center',fontsize=8)
save(fig,'figure_2')
macro=pd.read_csv(V2/'tables/subject_macro_metrics.csv',dtype={'seed':str})
fig,axes=plt.subplots(1,2,figsize=(8,3.6))
for ax,cohort in zip(axes,['ddatshr','diadata']):
    for model,g in macro[macro.dataset.eq(cohort)&macro.seed.eq('mean')&macro.condition.eq('cgm_hr')&~macro.model.isin(['LinearTrend','TreeStack'])].groupby('model'):
        g=g.sort_values('horizon');ax.plot(g.horizon,g.MAE,'o-',ms=3,label=LABEL.get(model,model))
    ax.set(title=LABEL[cohort],xlabel='Forecast horizon (minutes)',ylabel='Subject-macro MAE (mg/dL)',xticks=[30,60,90]);ax.grid(alpha=.2)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.16),ncol=4,fontsize=8,frameon=False)
save(fig,'figure_3')
profiles=pd.read_csv(V2/'tables/patient_profiles.csv',dtype={'PtID':str})
fig,axes=plt.subplots(1,2,figsize=(8,3.3));g=profiles[profiles.dataset.eq('ddatshr')].sort_values('RMSE')
axes[0].bar(range(len(g)),g.RMSE,color=['#CC3311' if p=='907' else '#4477AA' for p in g.PtID]);axes[0].set(xticks=range(len(g)),xticklabels=g.PtID,xlabel='DDATSHR participant',ylabel='RF 30-minute RMSE (mg/dL)');axes[0].tick_params(axis='x',rotation=90)
for cohort,color in [('ddatshr','#4477AA'),('diadata','#228833')]:
    g=profiles[profiles.dataset.eq(cohort)];axes[1].scatter(g.sd_glucose,g.RMSE,label=LABEL[cohort],s=20,color=color)
axes[1].set(xlabel='Eligible-input glucose SD (mg/dL)',ylabel='RF 30-minute RMSE (mg/dL)');axes[1].legend(frameon=False)
r=profiles[profiles.dataset.eq('ddatshr')&profiles.PtID.eq('907')].iloc[0];axes[1].annotate('907',(r.sd_glucose,r.RMSE),xytext=(5,5),textcoords='offset points')
save(fig,'figure_4')
comp=pd.read_csv(V2/'tables/primary_comparisons.csv')
fig,axes=plt.subplots(1,3,figsize=(9,3.6))
for ax,key,title in zip(axes,['HR_added','Stack_vs_RF','RF_vs_persistence'],['RF with HR minus CGM','Stack minus RF','RF minus persistence']):
    g=comp[comp.comparison.eq(key)].sort_values(['dataset','horizon']);y=np.arange(len(g))
    ax.axvline(0,color='grey',lw=.8);ax.errorbar(g.difference_MAE,y,xerr=[g.difference_MAE-g.ci95_low,g.ci95_high-g.difference_MAE],fmt='o',capsize=3)
    ax.set(yticks=y,yticklabels=[f'{LABEL[r.dataset]} {r.horizon} min' for r in g.itertuples()],title=title,xlabel='MAE difference (mg/dL)');ax.invert_yaxis()
save(fig,'figure_6')
ext=pd.read_csv(V3/'tables/subject_macro_metrics.csv',dtype={'seed':str});c=pd.read_csv(V3/'tables/descriptive_comparisons.csv')
fig,axes=plt.subplots(1,2,figsize=(8,3.5))
for model in ['Persistence','RandomForest','BiLSTMTransformer','Stack']:
    g=ext[ext.dataset.eq('azt1d')&ext.seed.eq('mean')&ext.model.eq(model)].sort_values('horizon');axes[0].plot(g.horizon,g.MAE,'o-',label=LABEL.get(model,model))
axes[0].set(title='A  Frozen DDATSHR to AZT1D',xlabel='Forecast horizon (minutes)',ylabel='Macro MAE (mg/dL)',xticks=[30,60,90]);axes[0].legend(fontsize=7,frameon=False)
for cohort,offset in [('ddatshr',-2),('diadata',2)]:
    g=c[c.dataset.eq(cohort)&c.comparison.eq('HR_added_three_validation')].sort_values('horizon');axes[1].errorbar(g.horizon+offset,g.difference_MAE,yerr=[g.difference_MAE-g.ci95_low,g.ci95_high-g.difference_MAE],fmt='o',capsize=3,label=LABEL[cohort])
axes[1].axhline(0,color='grey',lw=.8);axes[1].set(title='B  RF HR effect with three validators',xlabel='Forecast horizon (minutes)',ylabel='Paired MAE difference (mg/dL)',xticks=[30,60,90]);axes[1].legend(fontsize=8,frameon=False)
save(fig,'figure_7')
(OUT/'provenance.json').write_text(json.dumps({'evidence':'historical saved summaries; no new training or inference','files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [V2/'llm/generation_metrics.csv',V2/'llm/guardrail_challenge.csv',V2/'tables/subject_macro_metrics.csv',V2/'tables/patient_profiles.csv',V2/'tables/primary_comparisons.csv',V3/'tables/subject_macro_metrics.csv',V3/'tables/descriptive_comparisons.csv']}},indent=2))
