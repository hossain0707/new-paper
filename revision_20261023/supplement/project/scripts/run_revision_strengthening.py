"""Run additional experiments without changing or overwriting the primary evaluation."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,concurrent.futures,multiprocessing as mp,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.revision.strengthening import fit
from src.revision.experiment import BASE,SEEDS

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['external','validation'],required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    tasks=[];out=ROOT/'results/revision_v3';out.mkdir(exist_ok=True)
    for dataset in (['ddatshr'] if a.mode=='external' else ['ddatshr','diadata']):
        meta=pd.read_parquet(ROOT/f'data/processed/revision_v2/history_120/{dataset}_samples.parquet')
        for held in (['development'] if a.mode=='external' else sorted(meta.PtID.astype(str).unique())):
            for condition in (['glucose_only'] if a.mode=='external' else ['cgm_hr','cgm']):
                for model in (BASE if a.mode=='external' else ['RandomForest']):
                    for seed in SEEDS:
                        tasks.append((str(ROOT),str(out),a.mode,dataset,held,model,seed,condition,len(tasks)%4))
    print('planned',len(tasks),'fits',flush=True)
    with concurrent.futures.ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
        for f in concurrent.futures.as_completed([pool.submit(fit,t) for t in tasks]):print(f.result(),flush=True)
    print('COMPLETE',len(tasks),flush=True)
if __name__=='__main__':main()
