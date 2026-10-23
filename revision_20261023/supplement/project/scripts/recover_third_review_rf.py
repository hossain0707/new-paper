"""Regenerate missing historical RF seed-42 artifacts on checksum-matched data.

No historical artifact is overwritten. Verify every per-person MAE against the
saved seed-42 table before allowing case building. Parallel CPU fits only.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,concurrent.futures,json,multiprocessing,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.revision.experiment import run_task
from src.revision.data import sha256

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42])
    parser.add_argument('--output',type=Path,default=ROOT/'results/revision_v4/recovered_rf_seed42');args=parser.parse_args()
    assert set(args.seeds)<=set([42,1729,2026])
    out=args.output;assert 'revision_v4' in out.parts;out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/processed/revision_v2/history_120'
    expected={'ddatshr':'cbe727f2edf45ab0145774e03f235f34d2fa7c13c82d49c340c98a3d0458c7d9',
              'diadata':'54a12bfb016944a999ef51be91f749ba48c51b95c895b3a3d6bea1f893a4a35d'}
    tasks=[]
    for dataset in expected:
        assert sha256(data/f'{dataset}.npz')==expected[dataset]
        meta=pd.read_parquet(data/f'{dataset}_samples.parquet')
        for pt in sorted(meta.PtID.astype(str).unique()):
            for seed in args.seeds:tasks.append((str(data),str(out),dataset,pt,'RandomForest',seed,'cgm_hr',0,80))
    with concurrent.futures.ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run_task,t) for t in tasks]):print(json.dumps(f.result()),flush=True)
    old=pd.read_csv(ROOT/'results/revision_v2/tables/per_patient_metrics.csv',dtype={'seed':str,'PtID':str})
    old=old[old.seed.isin([str(s) for s in args.seeds])&old.model.eq('RandomForest')&old.condition.eq('cgm_hr')]
    audit=[]
    for task in tasks:
        dataset,pt=task[2:4];seed=task[5]
        frame=pd.read_parquet(out/f'fits/{dataset}/cgm_hr/RandomForest/seed_{seed}/{pt}/test_predictions.parquet')
        for h in (30,60,90):
            measured=float(np.abs(frame[f'true_{h}']-frame[f'pred_{h}']).mean())
            saved=old[old.dataset.eq(dataset)&old.PtID.eq(pt)&old.horizon.eq(h)&old.seed.eq(str(seed))].iloc[0]
            audit.append({'dataset':dataset,'PtID':pt,'seed':seed,'horizon':h,'recomputed_MAE':measured,'saved_MAE':float(saved.MAE),'difference':measured-saved.MAE,'n':len(frame)})
            assert len(frame)==saved.n
            assert abs(measured-saved.MAE)<1e-7,(dataset,pt,h,measured,saved.MAE)
    pd.DataFrame(audit).to_csv(out/'historical_metric_verification.csv',index=False)
    (out/'VERIFIED.json').write_text(json.dumps({'fits':len(tasks),'per_person_horizon_checks':len(audit),'tolerance_MAE':1e-7,'arrays':expected},indent=2))
    print('RECOVERED_RF_VERIFIED',flush=True)

if __name__=='__main__':main()
