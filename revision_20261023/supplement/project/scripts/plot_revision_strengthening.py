"""Additional evidence figure from saved participant-level summaries."""
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/revision_v3';dest=out/'figures';dest.mkdir(exist_ok=True)
m=pd.read_csv(out/'tables/subject_macro_metrics.csv',dtype={'seed':str});m=m[m.seed=='mean'];c=pd.read_csv(out/'tables/descriptive_comparisons.csv')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(7.0,5.6),layout='constrained')
grid=fig.add_gridspec(2,2)
axs=[fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1])]
for name,color in [('Persistence','#666666'),('RandomForest','#0072B2'),('BiLSTMTransformer','#009E73'),('Stack','#D55E00')]:
 d=m[(m.dataset=='azt1d')&(m.model==name)].sort_values('horizon');axs[0].plot(d.horizon,d.MAE,'o-',label=name,color=color)
axs[0].set(title='A  Frozen DDATSHR → AZT1D',xlabel='Forecast horizon (min)',ylabel='Macro MAE (mg/dL)',xticks=[30,60,90]);axs[0].legend(fontsize=8,ncol=2)
for dataset,color,offset in [('ddatshr','#0072B2',-2),('diadata','#D55E00',2)]:
 for ax,contrast,title in [(axs[1],'Chronos2_vs_RandomForest_cgm','B  Chronos-2 − CGM RF'),(axs[2],'HR_added_three_validation','C  HR effect: three validators')]:
  d=c[(c.dataset==dataset)&(c.comparison==contrast)].sort_values('horizon');ax.errorbar(d.horizon+offset,d.difference_MAE,yerr=np.array([d.difference_MAE-d.ci95_low,d.ci95_high-d.difference_MAE]),fmt='o',capsize=3,label=dataset,color=color)
  ax.axhline(0,color='grey',lw=.8);ax.set(title=title,xlabel='Forecast horizon (min)',ylabel='Paired MAE difference (mg/dL)',xticks=[30,60,90]);ax.legend(fontsize=8)
for ext in ['png','svg','pdf']:fig.savefig(dest/f'fig7_additional_evidence.{ext}',dpi=600)
