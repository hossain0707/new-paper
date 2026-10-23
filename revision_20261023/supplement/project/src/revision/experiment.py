"""Patient-disjoint nested-holdout training with resumable, fully identified artifacts."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

SEEDS = (42, 1729, 2026)
TREE_GRID = {'Ridge': [{'alpha':a} for a in (1.,10.,100.)],
             'RandomForest': [{'max_depth':d} for d in (8,16)],
             'XGBoost': [{'max_depth':d} for d in (3,6)]}
DEEP = ('TFTEncoder','BiLSTMTransformer','CNNGRU')
BASE = (*TREE_GRID.keys(), *DEEP)


def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def split_subjects(meta, held_out, seed=712):
    """Split independent of training seed, so comparisons share all subject allocations."""
    subjects=sorted(set(meta.PtID.astype(str))-{str(held_out)})
    digest=int(hashlib.sha256(str(held_out).encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed+digest);rng.shuffle(subjects)
    return subjects[1:],subjects[0],str(held_out)


def weights(subjects):
    unique,count=np.unique(subjects,return_counts=True)
    lookup=dict(zip(unique,count));w=np.array([1/lookup[s] for s in subjects])
    return w/w.mean()


def make_tree(name,cfg,seed):
    if name=='Ridge':return Ridge(**cfg)
    if name=='RandomForest':return RandomForestRegressor(n_estimators=300,min_samples_leaf=5,
                                                        n_jobs=1,random_state=seed,**cfg)
    return xgb.XGBRegressor(n_estimators=300,learning_rate=.05,subsample=.8,colsample_bytree=.8,
                            n_jobs=1,random_state=seed,**cfg)


def make_deep(name,n_channels):
    from src.models.deep_models import TFT,BiLSTMTransformer,CNNGRU
    if name=='TFTEncoder':return TFT(n_channels),True
    if name=='BiLSTMTransformer':return BiLSTMTransformer(n_channels),False
    if n_channels==4:return CNNGRU([0,2,3],[1],[]),False
    # CGM-only CNN ablation has one modality, not an artificial duplicated HR stream.
    import torch.nn as nn
    class SingleCNNGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=nn.Sequential(nn.Conv1d(n_channels,32,5,padding=2),nn.ReLU(),
                                     nn.Conv1d(32,64,3,padding=1),nn.ReLU())
            self.gru=nn.GRU(64,64,2,batch_first=True,dropout=.1);self.head=nn.Linear(64,3)
        def forward(self,x):
            z=self.conv(x.transpose(1,2)).transpose(1,2);z,_=self.gru(z);return self.head(z[:,-1])
    return SingleCNNGRU(),False


def fit_deep(name,xt,yt,xv,yv,w,seed,device,logdir,max_epochs=80):
    import torch
    torch.set_num_threads(1)
    mu=xt.mean(axis=(0,1));sd=xt.std(axis=(0,1))+1e-6
    ym=yt.mean(axis=0);ys=yt.std(axis=0)+1e-6
    tx=torch.tensor((xt-mu)/sd,device=device);ty=torch.tensor((yt-ym)/ys,device=device)
    vx=torch.tensor((xv-mu)/sd,device=device);tw=torch.tensor(w,dtype=torch.float32,device=device)
    winner=None;best_score=float('inf');histories=[]
    for lr in (.001,.0003):
        seed_all(seed)
        model,isq=make_deep(name,xt.shape[-1]);model=model.to(device)
        opt=torch.optim.Adam(model.parameters(),lr=lr)
        best=float('inf');stale=0;state=None;best_epoch=0
        for epoch in range(1,max_epochs+1):
            model.train();order=torch.randperm(len(tx),device=device);sumloss=0.
            for indices in order.split(1024):
                opt.zero_grad(set_to_none=True);out=model(tx[indices]);target=ty[indices]
                if isq:
                    err=target.unsqueeze(-1)-out;q=torch.tensor([.1,.5,.9],device=device)
                    individual=torch.maximum(q*err,(q-1)*err).mean(dim=(1,2))
                else:individual=(out-target).square().mean(dim=1)
                loss=(individual*tw[indices]).mean();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                sumloss+=float(loss.detach())*len(indices)
            model.eval();pieces=[]
            with torch.no_grad():
                for batch in vx.split(2048):
                    out=model(batch);pieces.append((out[:,:,1] if isq else out).cpu().numpy())
            yp=np.concatenate(pieces)*ys+ym;score=float(np.abs(yp-yv).mean())
            histories.append({'lr':lr,'epoch':epoch,'train_objective':sumloss/len(tx),'validation_MAE':score})
            if score<best-1e-5:
                best=score;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch
            else:stale+=1
            if stale>=10:break
        if best<best_score:
            best_score=best;winner={'state_dict':state,'feature_mean':mu,'feature_std':sd,
                                  'target_mean':ym,'target_std':ys,'lr':lr,'epoch':best_epoch,'is_quantile':isq}
        del model;torch.cuda.empty_cache()
    pd.DataFrame(histories).to_csv(logdir/'training_history.csv',index=False)
    torch.save(winner,logdir/'checkpoint.pt')
    model,isq=make_deep(name,xt.shape[-1]);model.load_state_dict(winner['state_dict']);model.to(device).eval()
    def predict(x):
        predictions=[]
        with torch.no_grad():
            for start in range(0,len(x),2048):
                z=torch.tensor((x[start:start+2048]-mu)/sd,device=device)
                out=model(z);predictions.append((out[:,:,1] if isq else out).cpu().numpy())
        return np.concatenate(predictions)*ys+ym
    return predict,{'lr':winner['lr'],'epoch':winner['epoch'],'inner_MAE':best_score}


def run_task(task):
    """One patient/model/seed/feature condition; saves before marking completion."""
    data_dir,out_dir,dataset,held_out,name,seed,condition,gpu,max_epochs=task
    path=Path(out_dir)/'fits'/dataset/condition/name/f'seed_{seed}'/str(held_out)
    from src.revision.data import sha256
    identity={'array_sha256':sha256(Path(data_dir)/f'{dataset}.npz'),
              'training_code_sha256':sha256(Path(__file__)), 'max_epochs':max_epochs}
    if (path/'complete.json').exists():
        prior=json.loads((path/'complete.json').read_text())
        if prior.get('identity')!=identity:raise RuntimeError(f'Stale artifacts: {path}; use a new output directory')
        return {'cached':True,'path':str(path)}
    path.mkdir(parents=True,exist_ok=True)
    started=time.time()
    data=np.load(Path(data_dir)/f'{dataset}.npz');X=data['X'];Y=data['y']
    meta=pd.read_parquet(Path(data_dir)/f'{dataset}_samples.parquet');meta.PtID=meta.PtID.astype(str)
    train,val,test=split_subjects(meta,held_out)
    tm=meta.PtID.isin(train).to_numpy();vm=meta.PtID.eq(val).to_numpy();sm=meta.PtID.eq(test).to_numpy()
    current=X[:,-1,0].copy();delta=Y-current[:,None]
    if condition=='cgm':X=X[:,:,[0,2,3]]
    w=weights(meta.PtID.to_numpy()[tm])
    if name in TREE_GRID:
        flat=X.reshape(len(X),-1)
        scaler=StandardScaler().fit(flat[tm]);z=scaler.transform(flat)
        best_score=float('inf');search=[]
        for cfg in TREE_GRID[name]:
            model=make_tree(name,cfg,seed);model.fit(z[tm],delta[tm],sample_weight=w)
            score=float(np.abs(model.predict(z[vm])-delta[vm]).mean());search.append({**cfg,'validation_MAE':score})
            if score<best_score:best_score=score;best_model=model;chosen=cfg
        pred_val=best_model.predict(z[vm])+current[vm,None]
        pred_test=best_model.predict(z[sm])+current[sm,None]
        joblib.dump({'model':best_model,'scaler':scaler,'condition':condition,'shape':X.shape[1:]},path/'checkpoint.joblib',compress=3)
        pd.DataFrame(search).to_csv(path/'search.csv',index=False)
        cfg={**chosen,'inner_MAE':best_score}
    else:
        predict,cfg=fit_deep(name,X[tm],delta[tm],X[vm],delta[vm],w,seed,f'cuda:{gpu}',path,max_epochs)
        pred_val=predict(X[vm])+current[vm,None];pred_test=predict(X[sm])+current[sm,None]
    for split,mask,pred in [('validation',vm,pred_val),('test',sm,pred_test)]:
        frame=meta.loc[mask].copy()
        for hi,h in enumerate((30,60,90)):
            frame[f'true_{h}']=Y[mask,hi];frame[f'pred_{h}']=pred[:,hi]
        frame.to_parquet(path/f'{split}_predictions.parquet',index=False)
    manifest={'identity':identity,'dataset':dataset,'test_subject':test,'validation_subject':val,'train_subjects':train,
              'model':name,'seed':seed,'condition':condition,'config':cfg,'train_n':int(tm.sum()),
              'validation_n':int(vm.sum()),'test_n':int(sm.sum()),'duration_s':time.time()-started}
    (path/'complete.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def simplex_weights(predictions,truth):
    """Direct constrained least squares; never normalize an unconstrained fit."""
    n=predictions.shape[1]
    result=minimize(lambda w:np.mean((predictions@w-truth)**2),np.ones(n)/n,
                    jac=lambda w:2*predictions.T@(predictions@w-truth)/len(truth),
                    bounds=[(0,1)]*n,constraints={'type':'eq','fun':lambda w:w.sum()-1,
                    'jac':lambda w:np.ones(n)},method='SLSQP',options={'maxiter':1000,'ftol':1e-9})
    if not result.success:raise RuntimeError(result.message)
    return result.x
