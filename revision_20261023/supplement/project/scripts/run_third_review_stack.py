"""Full five-member Stack with three validation people and patient balancing.

Only new revision-v4 artifacts are written. Pilot uses the lexicographically first
DDATSHR participant, seed 42, without inspecting test errors. No fitting on test.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,concurrent.futures,hashlib,json,multiprocessing,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.revision.strengthening import fit,weighted_simplex

MEMBERS=('RandomForest','XGBoost','TFTEncoder','BiLSTMTransformer','CNNGRU')
SEEDS=(42,1729,2026)

def fit_guarded(task):
    available_kb=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    if available_kb<8*1024*1024:raise RuntimeError('Less than 8 GiB available; defer fitting without stopping other services')
    return fit(task)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true')
    p.add_argument('--tree-workers',type=int,default=12)
    p.add_argument('--gpu-workers',type=int,default=2)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert 'revision_v4' in a.output.parts
    a.output.mkdir(parents=True,exist_ok=True)
    protocol={'post_review':True,'members':MEMBERS,'seeds':[42] if a.pilot else SEEDS,
        'conditions':['cgm_hr'],'inner_validation_people':3,'training_and_validation_patient_balanced':True,
        'weight_objective':'equal-participant squared error per horizon on validation only',
        'outer_origins':'unchanged history_120 metadata; verify hashes and sample IDs before interpretation',
        'tree_workers':a.tree_workers,'gpu_workers':a.gpu_workers,
        'pilot':a.pilot,'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [Path(__file__),ROOT/'src/revision/strengthening.py',ROOT/'src/revision/experiment.py',ROOT/'protocol_v4.md']}}
    frozen=a.output/'protocol.json'
    normalized=json.loads(json.dumps(protocol))
    if frozen.exists():assert json.loads(frozen.read_text())==normalized
    else:frozen.write_text(json.dumps(protocol,indent=2))
    if not a.pilot:
        tree_tasks=[];deep_tasks=[]
        for dataset in ['ddatshr','diadata']:
            meta=pd.read_parquet(ROOT/f'data/processed/revision_v2/history_120/{dataset}_samples.parquet')
            for person in sorted(meta.PtID.astype(str).unique()):
                for seed in SEEDS:
                    for model in ['RandomForest','XGBoost']:
                        tree_tasks.append((str(ROOT),str(a.output),'validation',dataset,person,model,seed,'cgm_hr',0))
                    for model in ['TFTEncoder','BiLSTMTransformer','CNNGRU']:
                        deep_tasks.append((str(ROOT),str(a.output),'validation',dataset,person,model,seed,'cgm_hr',0))
        with concurrent.futures.ProcessPoolExecutor(a.tree_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in concurrent.futures.as_completed([pool.submit(fit,t) for t in tree_tasks]):
                print('TREE_COMPLETE',future.result(),flush=True)
        with concurrent.futures.ProcessPoolExecutor(a.gpu_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in concurrent.futures.as_completed([pool.submit(fit_guarded,t) for t in deep_tasks]):
                print('NEURAL_COMPLETE',future.result(),flush=True)
    for dataset in (['ddatshr'] if a.pilot else ['ddatshr','diadata']):
        meta=pd.read_parquet(ROOT/f'data/processed/revision_v2/history_120/{dataset}_samples.parquet')
        people=sorted(meta.PtID.astype(str).unique())
        for person in (people[:1] if a.pilot else people):
            for seed in ([42] if a.pilot else SEEDS):
                frames={}
                for model in MEMBERS:
                    available_kb=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
                    if available_kb<8*1024*1024:raise RuntimeError('Less than 8 GiB available; defer fitting without stopping other services')
                    start=time.time()
                    folder=Path(fit((str(ROOT),str(a.output),'validation',dataset,person,model,seed,'cgm_hr',0)))
                    frames[model]={s:pd.read_parquet(folder/f'{s}.parquet').sort_values('sample_id').reset_index(drop=True) for s in ['validation','test']}
                    print(json.dumps({'dataset':dataset,'person':person,'model':model,'seed':seed,'elapsed_s':time.time()-start}),flush=True)
                reference=frames[MEMBERS[0]]
                output=reference['test'].copy();weights={}
                for h in (30,60,90):
                    for s in ['validation','test']:
                        for model in MEMBERS:
                            assert frames[model][s].sample_id.equals(reference[s].sample_id)
                            np.testing.assert_array_equal(frames[model][s][f'true_{h}'],reference[s][f'true_{h}'])
                    vp=np.column_stack([frames[m]['validation'][f'pred_{h}'] for m in MEMBERS])/100
                    w=weighted_simplex(vp,reference['validation'][f'true_{h}'].to_numpy()/100,reference['validation'].PtID.to_numpy())
                    weights[str(h)]=w.tolist()
                    output[f'pred_{h}']=np.column_stack([frames[m]['test'][f'pred_{h}'] for m in MEMBERS])@w
                target=a.output/'validation/stack'/dataset/f'seed_{seed}'/person
                target.mkdir(parents=True,exist_ok=True)
                output.to_parquet(target/'test.parquet',index=False)
                (target/'weights.json').write_text(json.dumps({'members':MEMBERS,'weights':weights},indent=2))
    print('PILOT_COMPLETE' if a.pilot else 'FULL_STACK_COMPLETE',flush=True)

if __name__=='__main__':main()
