import sys,json,hashlib
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(sys.argv[1]);OUT=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=True)
raw=ROOT/'results/revision_v4/rf_raw_verification/rf_seed_mean_predictions.parquet'
predictions=pd.read_parquet(raw)
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'figure_5.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
def main():
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.8),layout='constrained');densities=[]
    for ax,dataset in zip(axes,['ddatshr','diadata']):
        d=predictions[predictions.dataset==dataset]
        densities.append(ax.hexbin(d.true_30,d.pred_30,gridsize=55,mincnt=1,cmap='Blues',bins='log',extent=(0,450,0,450),rasterized=True))
        # Complete Clarke boundary geometry, including lower C and D segments.
        segments=[([0,450],[0,450]),([0,175/3],[70,70]),([175/3,375],[70,450]),([70,70],[0,56]),
                  ([70,450],[56,360]),([0,70],[180,180]),([70,70],[180,450]),([70,290],[180,400]),([290,290],[400,450]),
                  ([130,180],[0,70]),([180,180],[0,70]),([180,450],[70,70]),
                  ([240,240],[70,180]),([240,450],[180,180])]
        for i,(x,y) in enumerate(segments):ax.plot(x,y,'--' if i==0 else '-',c='#444444',lw=.65)
        for x,y,label in [(110,115,'A'),(210,300,'B'),(140,390,'C'),(150,15,'C'),(330,120,'D'),(30,120,'D'),(330,30,'E'),(25,320,'E')]:
            ax.text(x,y,label,color='#AA3311',fontsize=10)
        ax.set(xlim=(0,450),ylim=(0,450),aspect='equal',xlabel='Observed target (mg/dL)',ylabel='30-minute forecast (mg/dL)',title={'ddatshr':'DDATSHR','diadata':'DiaData'}[dataset])
    from matplotlib.colors import LogNorm
    norm=LogNorm(vmin=1,vmax=max(float(d.get_array().max()) for d in densities))
    for d in densities:d.set_norm(norm)
    fig.colorbar(densities[0],ax=axes,label='Origins per hexagon (log scale)',shrink=.7,pad=.02)
    save(fig,'fig5_clarke')

main()
(OUT/'figure5_provenance.json').write_text(json.dumps({'raw_prediction_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'pairs':len(predictions),'reconstructed_RF_seed_mean':True},indent=2))
