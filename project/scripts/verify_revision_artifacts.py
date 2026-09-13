"""Independent integrity checks of split identity, predictions, checkpoints and report inputs."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from src.revision.data import sha256
from src.revision.experiment import BASE,SEEDS,make_deep,seed_all
from src.revision.grounding import validate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--forecasts-only',action='store_true');parser.add_argument('--final-only',action='store_true');args=parser.parse_args()
    out=ROOT/'results/revision_v2';checks=[];total=0
    if not args.final_only:
        for dataset in ('ddatshr','diadata'):
            data=ROOT/'data/processed/revision_v2/history_120';arr=np.load(data/f'{dataset}.npz')
            meta=pd.read_parquet(data/f'{dataset}_samples.parquet');meta.PtID=meta.PtID.astype(str)
            source=meta.assign(array_index=np.arange(len(meta))).set_index('sample_id');expected_hash=sha256(data/f'{dataset}.npz')
            for condition in ('cgm_hr','cgm'):
                for model in BASE:
                    for seed in SEEDS:
                        ids=[]
                        for pt in sorted(meta.PtID.unique()):
                            path=out/'fits'/dataset/condition/model/f'seed_{seed}'/pt
                            manifest=json.loads((path/'complete.json').read_text());total+=1
                            assert manifest['identity']['array_sha256']==expected_hash
                            assert manifest['identity']['training_code_sha256']==sha256(ROOT/'src/revision/experiment.py')
                            assert pt not in manifest['train_subjects'] and manifest['validation_subject'] not in manifest['train_subjects']
                            assert pt!=manifest['validation_subject']
                            train=set(manifest['train_subjects']);assert train|{pt,manifest['validation_subject']}==set(meta.PtID)
                            for split in ('test','validation'):
                                f=pd.read_parquet(path/f'{split}_predictions.parquet')
                                assert f.PtID.astype(str).nunique()==1
                                assert str(f.PtID.iloc[0])==(pt if split=='test' else manifest['validation_subject'])
                                indices=source.loc[f.sample_id,'array_index'].to_numpy()
                                np.testing.assert_array_equal(f[[f'true_{h}' for h in (30,60,90)]],arr['y'][indices])
                                assert np.isfinite(f[[f'pred_{h}' for h in (30,60,90)]]).all().all()
                                if split=='test':ids.extend(f.sample_id)
                            # Reload each architecture and feature condition once per cohort.
                            if seed==42 and pt==sorted(meta.PtID.unique())[0]:
                                f=pd.read_parquet(path/'test_predictions.parquet').iloc[:2048]
                                ix=source.loc[f.sample_id,'array_index'].to_numpy();x=arr['X'][ix];current=x[:,-1,0]
                                if condition=='cgm':x=x[:,:,[0,2,3]]
                                if (path/'checkpoint.joblib').exists():
                                    saved=joblib.load(path/'checkpoint.joblib')
                                    prediction=saved['model'].predict(saved['scaler'].transform(x.reshape(len(x),-1)))+current[:,None]
                                else:
                                    import torch
                                    torch.set_num_threads(2);seed_all(seed)
                                    saved=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
                                    net,isq=make_deep(model,x.shape[-1]);net.load_state_dict(saved['state_dict']);net.to('cuda:0').eval()
                                    with torch.no_grad():
                                        z=net(torch.tensor((x-saved['feature_mean'])/saved['feature_std'],device='cuda:0'))
                                        z=z[:,:,1] if isq else z
                                    prediction=z.cpu().numpy()*saved['target_std']+saved['target_mean']+current[:,None]
                                np.testing.assert_allclose(prediction,f[[f'pred_{h}' for h in (30,60,90)]],atol=1e-4,rtol=1e-6,err_msg=str(path))
                        assert len(ids)==len(set(ids))==len(meta) and set(ids)==set(meta.sample_id)
            checks.append(f'{dataset}: all patient/model/seed/condition splits and targets matched; representative checkpoints reproduced predictions')
        assert total==1656
        weights=pd.read_csv(out/'tables/stack_weights.csv')
        assert weights.weight.between(-1e-8,1+1e-8).all()
        np.testing.assert_allclose(weights.groupby(['dataset','condition','PtID','seed','model','horizon']).weight.sum(),1,atol=1e-6)
        predictions=pd.read_parquet(out/'all_predictions.parquet')
        metrics=pd.read_csv(out/'tables/per_patient_metrics.csv',dtype={'PtID':str,'seed':str})
        indexed=metrics.set_index(['dataset','condition','model','seed','PtID','horizon'])
        for key,f in predictions.groupby(['dataset','condition','model','seed','PtID']):
            for h in (30,60,90):
                r=indexed.loc[(*key,h)]
                err=f[f'pred_{h}'].to_numpy(dtype=float)-f[f'true_{h}'].to_numpy(dtype=float)
                assert len(err)==r.n
                assert abs(np.abs(err).mean()-r.MAE)<1e-4 and abs(np.sqrt(np.square(err).mean())-r.RMSE)<1e-4
        checks.append('All per-patient MAE/RMSE cells independently recalculated from saved predictions')
        phase={'checks':checks,'primary_fits':total,'status':'passed',
               'prediction_sha256':sha256(out/'all_predictions.parquet'),
               'metrics_sha256':sha256(out/'tables/per_patient_metrics.csv'),
               'experiment_sha256':sha256(ROOT/'src/revision/experiment.py')}
        (out/'forecast_verification.json').write_text(json.dumps(phase,indent=2))
    else:
        phase=json.loads((out/'forecast_verification.json').read_text())
        assert phase['status']=='passed'
        assert phase['prediction_sha256']==sha256(out/'all_predictions.parquet')
        assert phase['metrics_sha256']==sha256(out/'tables/per_patient_metrics.csv')
        assert phase['experiment_sha256']==sha256(ROOT/'src/revision/experiment.py')
        checks=phase['checks'];total=phase['primary_fits']
    if args.forecasts_only:
        print(json.dumps(phase,indent=2));return
    selection=pd.read_csv(out/'llm/case_selection.csv',dtype={'PtID':str})
    assert not set(selection[selection.split=='test'].PtID)&set(selection[selection.split=='development'].PtID)
    responses=list((out/'llm/responses').glob('*.json'));assert len(responses)==3*len(selection)
    for p in responses:
        response=json.loads(p.read_text());case=json.loads((out/'llm/cases'/f"{response['case_id']}.json").read_text())
        v=validate(response['raw_text'],case);assert v['accepted']==response['validation']['accepted']
        expected_origin='deterministic_template' if response['arm']=='deterministic' else ('llm' if v['accepted'] else 'template_fallback')
        assert response['origin']==expected_origin
    freeze=json.loads((out/'llm/prompt_freeze.json').read_text())
    assert freeze['detector_sha256']==sha256(ROOT/'src/revision/grounding.py')
    assert freeze['selection_sha256']==sha256(out/'llm/case_selection.csv')
    checks.append('Patient-separated generation cases complete; stored gate decisions reproduce under frozen detector')
    manuscript=(ROOT/'manuscript/manuscript_REVISED_EVIDENCE.md').read_text();assert '{{' not in manuscript
    expected_title='Multimodal Deep Learning for Real-Time Glucose Prediction with Zero/Few-Shot On-Premises LLM Explainability: External Validation on an Independent Adolescent Cohort'
    assert manuscript.splitlines()[0]=='# '+expected_title
    import re
    assert re.findall(r'^## (\d+)\. ',manuscript,re.M)==[str(n) for n in range(1,10)]
    for section in ('7.1','7.2','7.3','7.4','7.5'):assert '### '+section+' ' in manuscript
    assert len(manuscript.split('## Abstract')[1].split('**Keywords')[0].split())<=200
    checks.append('Exact user-requested title, original nine-section structure and results sequence preserved; abstract within 200 words')
    vals=json.loads((out/'manuscript_values.json').read_text())
    for key in ('dd_rf_mae','dia_rf_mae','zero_accept','few_accept'):assert vals[key] in manuscript
    from docx import Document
    doc=Document(ROOT.parent/'manuscript_REVISED_EVIDENCE.docx');assert len(doc.inline_shapes)==7
    assert len(doc.tables)==8
    checks.append('Generated manuscript resolved all numerical slots and contains eight tables and seven figures')
    record={'status':'passed','primary_fits':total,'checks':checks,
            'limitations':'Integrity validation is not human clinical review, editorial approval or a proof of scientific validity.'}
    (out/'artifact_verification.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':main()
