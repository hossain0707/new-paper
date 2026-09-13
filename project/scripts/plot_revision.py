"""Publication figures generated solely from revision-v2 artifacts (PDF/SVG/600 dpi PNG)."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT=ROOT/'results/revision_v2';FIG=OUT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
COLORS=['#777777','#b89b72','#4477AA','#228833','#CCBB44','#AA3377','#66CCEE','#EE6677','#332288','#999933']
MODELS=['Persistence','LinearTrend','Ridge','RandomForest','XGBoost','TFTEncoder','BiLSTMTransformer','CNNGRU','Stack','TreeStack']


def save(fig,name):
    FIG.mkdir(exist_ok=True)
    for ext in ('pdf','svg','png'):fig.savefig(FIG/f'{name}.{ext}',dpi=600,bbox_inches='tight')
    plt.close(fig)


def main():
    macro=pd.read_csv(OUT/'tables/subject_macro_metrics.csv',dtype={'seed':str})
    pm=pd.read_csv(OUT/'tables/per_patient_metrics.csv',dtype={'PtID':str,'seed':str})
    fig,ax=plt.subplots(figsize=(7.2,4.3));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
    boxes=[(.3,4.4,4.1,1.1,'Observed CGM + heart rate\nDDATSHR original records\nDiaData raw integrated export'),
           (5.3,4.4,4.1,1.1,'Availability-aware histories\nNo local interpolation\nMatched observed targets'),
           (.3,2.6,4.1,1.1,'Patient-held-out forecasts\nShared inner validation patient\nTraining-only scaling'),
           (5.3,2.6,4.1,1.1,'Patient-level evaluation\nSeeds and source transfer\nHR and history ablations'),
           (.3,.6,4.1,1.2,'Held-out RF + genuine TreeSHAP\nFrozen narrative cases\nDevelopment/test split by patient'),
           (5.3,.6,4.1,1.2,'Local generation\nTyped validation and rendering\nCounted template fallback')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor='#edf3f8',edgecolor='#4477AA'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
    for p,q in [((4.5,4.95),(5.1,4.95)),((7.35,4.2),(2.35,3.9)),((4.5,3.15),(5.1,3.15)),((2.35,2.4),(2.35,2.0)),((4.5,1.2),(5.1,1.2))]:
        ax.annotate('',q,p,arrowprops={'arrowstyle':'->','color':'#555555','lw':1.4})
    save(fig,'fig1_protocol')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2))
    for ax,dataset in zip(axes,['ddatshr','diadata']):
        d=macro[(macro.dataset==dataset)&(macro.seed=='mean')&(macro.condition=='cgm_hr')]
        for model,color in zip(MODELS,COLORS):
            if model in ['LinearTrend','TreeStack']:continue
            a=d[d.model==model].sort_values('horizon');ax.plot(a.horizon,a.MAE,'o-',label=model,color=color,lw=1.3,ms=3)
        ax.set(title=dataset.upper(),xlabel='Forecast horizon (minutes)',ylabel='Subject-macro MAE (mg/dL)',xticks=[30,60,90]);ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.15),frameon=False,fontsize=8)
    fig.tight_layout();save(fig,'fig2_forecast_mae')
    comparisons=pd.read_csv(OUT/'tables/primary_comparisons.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.5,3.8),sharex=False)
    for ax,comparison in zip(axes,['HR_added','Stack_vs_RF','RF_vs_persistence']):
        d=comparisons[comparisons.comparison==comparison].sort_values(['dataset','horizon']).reset_index(drop=True)
        y=np.arange(len(d));ax.axvline(0,color='gray',lw=.8)
        ax.errorbar(d.difference_MAE,y,xerr=[d.difference_MAE-d.ci95_low,d.ci95_high-d.difference_MAE],fmt='o',color='#4477AA',capsize=3)
        ax.set(yticks=y,yticklabels=[f'{r.dataset} {r.horizon}m' for r in d.itertuples()],xlabel='MAE difference (mg/dL)',title=comparison.replace('_',' '))
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.tight_layout();save(fig,'fig3_paired_effects')
    profiles=pd.read_csv(OUT/'tables/patient_profiles.csv',dtype={'PtID':str})
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.3))
    d=profiles[profiles.dataset=='ddatshr'].sort_values('RMSE')
    axes[0].bar(np.arange(len(d)),d.RMSE,color=['#CC3311' if p=='907' else '#4477AA' for p in d.PtID])
    axes[0].set(xticks=np.arange(len(d)),xticklabels=d.PtID,ylabel='RF 30-minute RMSE (mg/dL)',xlabel='Held-out DDATSHR patient')
    axes[0].tick_params(axis='x',labelrotation=90)
    for dataset,color in [('ddatshr','#4477AA'),('diadata','#228833')]:
        d=profiles[profiles.dataset==dataset];axes[1].scatter(d.sd_glucose,d.RMSE,c=color,label=dataset,s=22,alpha=.8)
    r=profiles[(profiles.dataset=='ddatshr')&(profiles.PtID=='907')].iloc[0]
    axes[1].annotate('907',(r.sd_glucose,r.RMSE),xytext=(5,5),textcoords='offset points')
    axes[1].set(xlabel='Eligible-input glucose SD (mg/dL)',ylabel='RF 30-minute RMSE (mg/dL)');axes[1].legend(frameon=False)
    fig.tight_layout();save(fig,'fig4_heterogeneity')
    predictions=pd.read_parquet(OUT/'all_predictions.parquet')
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.8),layout='constrained');densities=[]
    for ax,dataset in zip(axes,['ddatshr','diadata']):
        d=predictions[(predictions.dataset==dataset)&(predictions.model=='RandomForest')&(predictions.seed=='mean')&(predictions.condition=='cgm_hr')]
        densities.append(ax.hexbin(d.true_30,d.pred_30,gridsize=55,mincnt=1,cmap='Blues',bins='log',extent=(0,450,0,450),rasterized=True))
        # Complete Clarke boundary geometry, including lower C and D segments.
        segments=[([0,450],[0,450]),([0,175/3],[70,70]),([175/3,375],[70,450]),([70,70],[0,56]),
                  ([70,450],[56,360]),([0,70],[180,180]),([70,70],[180,450]),([70,290],[180,400]),([290,290],[400,450]),
                  ([130,180],[0,70]),([180,180],[0,70]),([180,450],[70,70]),
                  ([240,240],[70,180]),([240,450],[180,180])]
        for i,(x,y) in enumerate(segments):ax.plot(x,y,'--' if i==0 else '-',c='#444444',lw=.65)
        for x,y,label in [(110,115,'A'),(210,300,'B'),(140,390,'C'),(150,15,'C'),(330,120,'D'),(30,120,'D'),(330,30,'E'),(25,320,'E')]:
            ax.text(x,y,label,color='#AA3311',fontsize=10)
        ax.set(xlim=(0,450),ylim=(0,450),aspect='equal',xlabel='Observed target (mg/dL)',ylabel='30-minute forecast (mg/dL)',title=dataset.upper())
    from matplotlib.colors import LogNorm
    norm=LogNorm(vmin=1,vmax=max(float(d.get_array().max()) for d in densities))
    for d in densities:d.set_norm(norm)
    fig.colorbar(densities[0],ax=axes,label='Origins per hexagon (log scale)',shrink=.7,pad=.02)
    save(fig,'fig5_clarke')
    generation=pd.read_csv(OUT/'llm/generation_metrics.csv');generation=generation[generation.split=='test']
    audit=pd.read_csv(OUT/'llm/guardrail_challenge.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2))
    arms=['deterministic','zero_shot','few_shot'];rates=[generation[generation.arm==a].accepted.mean()*100 for a in arms]
    axes[0].bar(arms,rates,color=['#777777','#4477AA','#228833']);axes[0].set(ylim=(0,110),ylabel='Complete contract acceptance (%)')
    for i,a in enumerate(arms):
        d=generation[generation.arm==a];axes[0].text(i,rates[i]+2,f'{int(d.accepted.sum())}/{len(d)}',ha='center',fontsize=8)
    groups=['control','development','challenge','boundary']
    rejection=[audit[audit.split==s].rejected.mean()*100 for s in groups]
    axes[1].bar(groups,rejection,color=['#777777','#4477AA','#228833','#CC3311']);axes[1].set(ylim=(0,110),ylabel='Rejection rate (%)')
    axes[1].tick_params(axis='x',rotation=20);fig.tight_layout();save(fig,'fig6_grounding')
    manifest={str(p.name):{'bytes':p.stat().st_size} for p in FIG.glob('*')}
    (FIG/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
