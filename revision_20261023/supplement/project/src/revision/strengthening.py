"""Additional, separately labelled evaluations; v2 training code is not modified."""
import os,copy,json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from src.revision.experiment import make_tree,make_deep,seed_all,weights,TREE_GRID,BASE,SEEDS
from src.revision.data import sha256


def patient_mae(pred,true,subjects):
    err=np.abs(np.asarray(pred)-np.asarray(true)).mean(axis=1)
    return float(pd.DataFrame({'subject':subjects,'error':err}).groupby('subject').error.mean().mean())


def grouped_split(meta,held_out):
    subjects=sorted(set(meta.PtID.astype(str))-{str(held_out)})
    digest=int(hashlib.sha256(str(held_out).encode()).hexdigest()[:8],16)
    np.random.default_rng(712+digest).shuffle(subjects)
    return subjects[3:],subjects[:3],[str(held_out)]


def weighted_simplex(pred,true,subjects):
    w=weights(subjects)
    n=pred.shape[1]
    result=minimize(lambda x:np.mean(w*(pred@x-true)**2),np.ones(n)/n,
      jac=lambda x:2*pred.T@(w*(pred@x-true))/len(true),bounds=[(0,1)]*n,
      constraints={'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones(n)},
      method='SLSQP',options={'maxiter':1000,'ftol':1e-9})
    if not result.success:raise RuntimeError(result.message)
    return result.x

def fit_deep_grouped(name,xt,yt,xv,yv,w,seed,device,logdir,val_subjects,max_epochs=80):
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
            yp=np.concatenate(pieces)*ys+ym;score=patient_mae(yp,yv,val_subjects)
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


def fit(task):
    root,out,mode,dataset,held_out,name,seed,condition,gpu=task
    root=Path(root);out=Path(out)
    data=root/'data/processed/revision_v2/history_120'
    array=np.load(data/f'{dataset}.npz');X=array['X'];Y=array['y']
    meta=pd.read_parquet(data/f'{dataset}_samples.parquet');meta.PtID=meta.PtID.astype(str)
    if mode=='external':
        subjects=sorted(meta.PtID.unique(),key=lambda p:hashlib.sha256(('v3_external|'+p).encode()).hexdigest())
        train,val,test=subjects[3:],subjects[:3],[];X=X[:,:,[0]]
    else:
        train,val,test=grouped_split(meta,held_out)
        if condition=='cgm':X=X[:,:,[0,2,3]]
    tm=meta.PtID.isin(train).to_numpy();vm=meta.PtID.isin(val).to_numpy();sm=meta.PtID.isin(test).to_numpy()
    assert not set(train)&set(val) and not set(train)&set(test) and not set(val)&set(test)
    path=out/mode/'fits'/dataset/condition/name/f'seed_{seed}'/str(held_out)
    identity={'data_sha256':sha256(data/f'{dataset}.npz'),'meta_sha256':sha256(data/f'{dataset}_samples.parquet'),
       'code_sha256':sha256(Path(__file__)),'base_code_sha256':sha256(root/'src/revision/experiment.py'),
       'deep_code_sha256':sha256(root/'src/models/deep_models.py'),'protocol_sha256':sha256(root/'protocol_v3.md'),
       'train':train,'validation':val,'test':test,'shape':list(X.shape[1:]),'seed':seed,'model':name,'condition':condition}
    if (path/'complete.json').exists():
        manifest=json.loads((path/'complete.json').read_text());assert manifest['identity']==identity
        assert (path/'validation.parquet').exists() and ((path/'checkpoint.pt').exists() or (path/'checkpoint.joblib').exists())
        if test:assert (path/'test.parquet').exists()
        return str(path)
    path.mkdir(parents=True,exist_ok=True);started=time.time()
    current=X[:,-1,0];delta=Y-current[:,None];vs=meta.PtID.to_numpy()[vm]
    if name in TREE_GRID:
        flat=X.reshape(len(X),-1);scaler=StandardScaler().fit(flat[tm]);z=scaler.transform(flat)
        best=float('inf');search=[]
        for cfg in TREE_GRID[name]:
            model=make_tree(name,cfg,seed);model.fit(z[tm],delta[tm],sample_weight=weights(meta.PtID.to_numpy()[tm]))
            score=patient_mae(model.predict(z[vm]),delta[vm],vs);search.append({**cfg,'validation_macro_MAE':score})
            if score<best:best=score;chosen=cfg;best_model=model
        saved={'model':best_model,'scaler':scaler,'shape':X.shape[1:],'condition':condition}
        joblib.dump(saved,path/'checkpoint.joblib',compress=3)
        pd.DataFrame(search).to_csv(path/'search.csv',index=False)
        predict=lambda x:best_model.predict(scaler.transform(x.reshape(len(x),-1)))
        config={**chosen,'validation_macro_MAE':best}
    else:
        predict,config=fit_deep_grouped(name,X[tm],delta[tm],X[vm],delta[vm],weights(meta.PtID.to_numpy()[tm]),seed,f'cuda:{gpu}',path,vs)
    for label,mask in [('validation',vm),('test',sm)]:
        if not mask.any():continue
        prediction=predict(X[mask])+current[mask,None];frame=meta.loc[mask].copy()
        for i,h in enumerate([30,60,90]):frame[f'true_{h}']=Y[mask,i];frame[f'pred_{h}']=prediction[:,i]
        assert np.isfinite(prediction).all();frame.to_parquet(path/f'{label}.parquet',index=False)
    manifest={'identity':identity,'config':config,'duration_s':time.time()-started,'train_n':int(tm.sum()),'validation_n':int(vm.sum()),'test_n':int(sm.sum())}
    (path/'complete.json').write_text(json.dumps(manifest,indent=2));return str(path)
