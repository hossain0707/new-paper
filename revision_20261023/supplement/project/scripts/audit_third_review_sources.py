"""Read-only source-ID and checksum audit after pinned source acquisition."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.revision.data import strict_csv,sha256

out=ROOT/'results/revision_v4/source_audit';out.mkdir(parents=True,exist_ok=True)
root=ROOT/'data/raw/ddatshr';audit={};counts={}
for name,col in [('Abbott.csv','Historic Glucose [mmol/l]'),('Medtronic.csv','Sensor Glucose [mmol/l]')]:
    d=strict_csv(root/name,audit)
    s=d[d['Subject code number'].astype(str).eq('973')]
    counts[name]={'rows_973':len(s),'numeric_selected_glucose_973':int(pd.to_numeric(s[col],errors='coerce').notna().sum())}
population=pd.read_csv(root/'population.csv')
ret=pd.read_csv(ROOT/'data/processed/revision_v2/history_120/ddatshr_retention.csv',dtype={'PtID':str})
dia_path=ROOT/'data/raw/diadata/SDBII_15min_sampling_raw.csv'
dia=pd.read_csv(dia_path);ids=sorted(dia[dia.Database.eq('DDATSHR')].PtID.unique())
normalized=sorted(str(int(float(p.split('_')[0]))) for p in ids)
summary={'source_population':int(population['Subject code number'].nunique()),
         'diadata_ddatshr_ids':ids,'normalized_source_ids':normalized,
         'primary_grid_ids':sorted(ret.PtID),'only_in_integrated_export':sorted(set(normalized)-set(ret.PtID)),
         'raw_selected_field_counts':counts,'diadata_raw_sha256':sha256(dia_path),
         'raw_original_files':audit,'script_sha256':sha256(Path(__file__)),
         'crosswalk_rule':'numeric part of explicit DDATSHR source ID; no inferred physiological identity',
         'arrays':{n:json.loads((ROOT/f'data/processed/revision_v2/history_120/{n}_audit.json').read_text())['array_sha256'] for n in ['ddatshr','diadata']}}
(out/'source_cohort_accounting.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
