"""Prepare observed CGM only, retaining gaps and auditing every public source file."""
import sys,json,zipfile,io,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.revision.data import sha256
p=ROOT/'data/raw/azt1d_v1/AZT1D_2025.zip'
assert sha256(p)=='2c6b4183fdc729c708c1b3230ab1988d2c9006dd35729537f2de07f649f0d113'
out=ROOT/'data/processed/revision_v3';out.mkdir(parents=True,exist_ok=True)
report=ROOT/'results/revision_v3/external';report.mkdir(parents=True,exist_ok=True)
xs=[];ys=[];meta=[];audit=[]
with zipfile.ZipFile(p) as z:
 names=sorted(n for n in z.namelist() if n.startswith('AZT1D 2025/CGM Records/') and n.endswith('.csv'))
 assert len(names)==25
 (ROOT/'results/revision_v3/provenance/azt1d_README.txt').write_bytes(z.read('AZT1D 2025/README.txt'))
 for name in names:
  pt='AZT'+re.search(r'Subject (\d+)\.csv$',name).group(1).zfill(2)
  raw=pd.read_csv(io.BytesIO(z.read(name)))
  if 'CGM' not in raw.columns:
   audit.append(dict(PtID=pt,file=name,raw_rows=len(raw),included=False,eligible_origins=0,exclusion_reason='No CGM-specific field; mixed Readings (CGM / BGM) cannot be disambiguated.'))
   continue
  t=pd.to_datetime(raw.EventDateTime,format='%Y-%m-%d %H:%M:%S',errors='coerce');g=pd.to_numeric(raw.CGM,errors='coerce')
  valid=t.notna()&g.notna()&np.isfinite(g)&(g>0)
  d=pd.DataFrame({'t':t[valid],'g':g[valid]}).sort_values('t',kind='stable')
  # Ambiguous duplicate timestamps are removed, rather than choosing a favorable reading.
  conflicts=d.groupby('t').g.nunique();bad=conflicts[conflicts>1].index
  d=d[~d.t.isin(bad)].drop_duplicates('t')
  s=d.set_index('t').g.resample('15min',closed='right',label='right').last()
  a=s.to_numpy();origins=np.arange(8,len(s)-6)
  xx=np.stack([a[origins-k] for k in range(8,-1,-1)],axis=1)
  yy=np.stack([a[origins+k] for k in [2,4,6]],axis=1)
  keep=np.isfinite(xx).all(axis=1)&np.isfinite(yy).all(axis=1)
  n=int(keep.sum());included=n>=50
  audit.append(dict(PtID=pt,file=name,raw_rows=len(raw),valid_cgm_rows=int(valid.sum()),invalid_timestamp_rows=int(t.isna().sum()),conflicting_timestamps=len(bad),grid_bins=len(s),observed_bins=int(s.notna().sum()),candidate_origins=len(origins),eligible_origins=n,included=included,first_time=str(s.index[0]),last_time=str(s.index[-1])))
  if included:
   times=s.index[origins[keep]];xs.append(xx[keep,:,None].astype('float32'));ys.append(yy[keep].astype('float32'))
   meta.extend(dict(PtID=pt,sample_id=f'{pt}|{stamp.isoformat()}',timestamp=stamp,source='AZT1D',input_lag_minutes=0) for stamp in times)
np.savez_compressed(out/'azt1d.npz',X=np.concatenate(xs),y=np.concatenate(ys))
pd.DataFrame(meta).to_parquet(out/'azt1d_samples.parquet',index=False)
pd.DataFrame(audit).to_csv(report/'retention.csv',index=False)
(report/'data_audit.json').write_text(json.dumps({'archive_sha256':sha256(p),'array_sha256':sha256(out/'azt1d.npz'),'metadata_sha256':sha256(out/'azt1d_samples.parquet'),'preparation_sha256':sha256(Path(__file__)),'patients':len(xs),'origins':len(meta),'timezone':'provider local Arizona; no conversion; no calendar covariates','interpolation':False,'clock_note':'Provider timestamps minute-resolved; last observed value in right-closed 15-minute bin.'},indent=2))
print('Prepared',len(xs),'patients',len(meta),'origins')
