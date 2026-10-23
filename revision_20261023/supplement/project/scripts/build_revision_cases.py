"""Fixed source/patient/state sampling and held-out TreeSHAP, with resumable case artifacts."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from src.revision.data import sha256
from src.revision.grounding import study_score


def main():
    p=argparse.ArgumentParser();p.add_argument('--select-only',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'results/revision_v2/llm')
    p.add_argument('--fit-root',type=Path,default=ROOT/'results/revision_v2')
    a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/processed/revision_v2/history_120';selected=[]
    for dataset in ('ddatshr','diadata'):
        m=pd.read_parquet(data/f'{dataset}_samples.parquet');m.PtID=m.PtID.astype(str)
        pts=sorted(m.PtID.unique(),key=lambda s:hashlib.sha256(s.encode()).hexdigest())
        dev=set(pts[:max(2,len(pts)//5)])
        for pt,d in m.groupby('PtID'):
            for state,rows in [('low',d[d.glucose_mgdl<70]),('in_range',d[d.glucose_mgdl.between(70,180)]),('high',d[d.glucose_mgdl>180])]:
                if len(rows):
                    r=rows.sample(1,random_state=2026).iloc[0]
                    selected.append({'dataset':dataset,'PtID':pt,'sample_id':r.sample_id,'state':state,
                                     'split':'development' if pt in dev else 'test','array_index':int(r.name)})
    selection=pd.DataFrame(selected);selection.to_csv(out/'case_selection.csv',index=False)
    if a.select_only:print(selection.groupby(['dataset','split']).size());return
    import shap
    folder=out/'cases';folder.mkdir(exist_ok=True);missing=[]
    for (dataset,pt),rows in selection.groupby(['dataset','PtID']):
        path=a.fit_root/'fits'/dataset/'cgm_hr/RandomForest/seed_42'/pt
        if not (path/'complete.json').exists():missing.append(dataset+'/'+pt);continue
        identifiers=[hashlib.sha256((dataset+'|'+sid).encode()).hexdigest()[:16] for sid in rows.sample_id]
        if all((folder/(identifier+'.json')).exists() for identifier in identifiers):continue
        indices=rows.array_index.to_numpy();arr=np.load(data/f'{dataset}.npz');x=arr['X'][indices]
        saved=joblib.load(path/'checkpoint.joblib');z=saved['scaler'].transform(x.reshape(len(x),-1))
        prediction=saved['model'].predict(z);explainer=shap.TreeExplainer(saved['model'])
        values=np.asarray(explainer.shap_values(z,check_additivity=True))
        if values.shape!=(len(x),x.shape[1]*4,3):raise ValueError(f'Unexpected SHAP dimensions: {values.shape}')
        np.testing.assert_allclose(values.sum(axis=1)+explainer.expected_value,prediction,atol=1e-4)
        for j,(_,r) in enumerate(rows.iterrows()):
            identifier=hashlib.sha256((dataset+'|'+r.sample_id).encode()).hexdigest()[:16]
            target=folder/(identifier+'.json')
            entries=[]
            for i in np.argsort(np.abs(values[j,:,0]))[-3:][::-1]:
                lag=(x.shape[1]-1-i//4)*15+(15 if dataset=='diadata' else 0)
                channel=['glucose_mgdl','hr_bpm','tod_sin','tod_cos'][i%4]
                entries.append({'feature':f'{channel}_lag_{lag}min','contribution_mgdl':round(float(values[j,i,0]),1)})
            glucose=round(float(x[j,-1,0]),1);rate=round(float((x[j,-1,0]-x[j,-2,0])/15),2)
            case={'case_id':identifier,**r.to_dict(),'model':'RandomForest seed 42; patient-held-out',
                  'checkpoint_sha256':sha256(path/'checkpoint.joblib'),
                  'values':{'observations':{'glucose_mgdl':glucose,'hr_bpm':round(float(x[j,-1,1]),1),
                                            'input_lag_minutes':15 if dataset=='diadata' else 0},
                            'forecasts_mgdl':{f'h{h}':round(float(prediction[j,k]+x[j,-1,0]),1) for k,h in enumerate((30,60,90))},
                            'rate_mgdl_per_min':rate,'study_score':study_score(glucose,rate),'shap_h30':entries},
                  'shap_audit':{'expected_delta':explainer.expected_value.tolist(),'full_values':values[j].tolist(),
                                'model_delta':prediction[j].tolist(),'feature_values':x[j].tolist()}}
            target.write_text(json.dumps(case,indent=2))
    print('cases',len(list(folder.glob('*.json'))),'selected',len(selection),'pending patients',missing)

if __name__=='__main__':main()
