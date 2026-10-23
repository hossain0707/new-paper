"""Secondary, separately labelled history and source-transfer experiments."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,concurrent.futures,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.revision.experiment import run_task,SEEDS
from src.evaluation.metrics import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['source','history','history_short_matched'],required=True);a=p.parse_args()
    out=ROOT/'results/revision_v2/sensitivity'/a.mode;out.mkdir(parents=True,exist_ok=True)
    tasks=[]
    if a.mode=='source':
        original=ROOT/'data/processed/revision_v2/history_120';m=pd.read_parquet(original/'diadata_samples.parquet')
        for source in sorted(m.source.unique()):
            data=ROOT/'data/processed/revision_v2/source_transfer'/source;data.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original/'diadata.npz',data/'diadata.npz')
            meta=m.copy();meta['original_PtID']=meta.PtID.astype(str);heldout='SOURCE_'+source
            meta.loc[meta.source.eq(source),'PtID']=heldout;meta.to_parquet(data/'diadata_samples.parquet',index=False)
            for model in ('RandomForest','XGBoost'):
                for seed in SEEDS:tasks.append((str(data),str(out),'diadata',heldout,model,seed,'cgm_hr',0,80))
    else:
        data=ROOT/'data/processed/revision_v2/history_240'
        for dataset in ('ddatshr','diadata'):
            meta=pd.read_parquet(data/f'{dataset}_samples.parquet')
            if a.mode=='history_short_matched':
                original=ROOT/'data/processed/revision_v2/history_120'
                sm=pd.read_parquet(original/f'{dataset}_samples.parquet')
                indices=sm.reset_index().set_index('sample_id').loc[meta.sample_id,'index'].to_numpy()
                arr=np.load(original/f'{dataset}.npz')
                matched=ROOT/'data/processed/revision_v2/history_120_matched_240';matched.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(matched/f'{dataset}.npz',X=arr['X'][indices],y=arr['y'][indices])
                meta.to_parquet(matched/f'{dataset}_samples.parquet',index=False)
                task_data=matched
            else:task_data=data
            for patient in sorted(meta.PtID.astype(str).unique()):
                for model in ('Ridge','XGBoost'):tasks.append((str(task_data),str(out),dataset,patient,model,42,'cgm_hr',0,80))
    with concurrent.futures.ProcessPoolExecutor(4) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run_task,t) for t in tasks]):print(json.dumps(f.result()),flush=True)
    rows=[]
    for path in out.glob('fits/*/*/*/*/*/test_predictions.parquet'):
        f=pd.read_parquet(path);manifest=json.loads((path.parent/'complete.json').read_text())
        if 'original_PtID' in f:f.PtID=f.original_PtID
        for pt,d in f.groupby('PtID'):
            for h in (30,60,90):rows.append({'dataset':manifest['dataset'],'model':manifest['model'],'seed':manifest['seed'],
                                            'PtID':pt,'horizon':h,'source':d.source.iloc[0],**summarize(d[f'true_{h}'],d[f'pred_{h}'])})
    pd.DataFrame(rows).to_csv(out/'per_patient_metrics.csv',index=False)

if __name__=='__main__':main()
