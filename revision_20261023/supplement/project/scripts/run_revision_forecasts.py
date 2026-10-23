"""Run corrected experiments, one process per GPU, resumably. No APIs called."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import concurrent.futures
import json
import multiprocessing as mp
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.revision.experiment import run_task, BASE, DEEP, SEEDS


def worker(tasks):
    for task in tasks:
        result=run_task(task)
        print(json.dumps(result),flush=True)
    return len(tasks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['ddatshr','diadata'])
    p.add_argument('--models',nargs='+',default=list(BASE));p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    p.add_argument('--conditions',nargs='+',default=['cgm_hr','cgm']);p.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3])
    p.add_argument('--workers-per-gpu',type=int,default=4)
    p.add_argument('--history',type=int,default=120);p.add_argument('--limit-subjects',type=int,default=0)
    p.add_argument('--max-epochs',type=int,default=80);p.add_argument('--output',default='results/revision_v2')
    a=p.parse_args();data=ROOT/'data/processed/revision_v2'/f'history_{a.history}';out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    from datetime import datetime, timezone
    (out/f'invocation_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")}.json').write_text(json.dumps(vars(a),indent=2))
    tasks=[]
    for d in a.datasets:
        meta=pd.read_parquet(data/f'{d}_samples.parquet');subjects=sorted(meta.PtID.astype(str).unique())
        if a.limit_subjects:subjects=subjects[:a.limit_subjects]
        for s in subjects:
            for condition in a.conditions:
                for model in a.models:
                    for seed in a.seeds:
                        tasks.append((str(data),str(out),d,s,model,seed,condition,a.gpus[len(tasks)%len(a.gpus)],a.max_epochs))
    nworkers=len(a.gpus)*a.workers_per_gpu
    bins=[tasks[i::nworkers] for i in range(nworkers)]
    print(f'Launching {len(tasks)} tasks across {len(bins)} GPU workers',flush=True)
    with concurrent.futures.ProcessPoolExecutor(len(bins),mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,b) for b in bins]
        for f in concurrent.futures.as_completed(futures):print('worker finished',f.result(),flush=True)


if __name__=='__main__':main()
