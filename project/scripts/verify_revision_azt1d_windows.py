"""Independently recover every prepared boundary from raw timestamps using searchsorted."""
import sys,zipfile,io,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from src.revision.data import sha256
arr=np.load(ROOT/'data/processed/revision_v3/azt1d.npz');meta=pd.read_parquet(ROOT/'data/processed/revision_v3/azt1d_samples.parquet');rows=[]
with zipfile.ZipFile(ROOT/'data/raw/azt1d_v1/AZT1D_2025.zip') as z:
 for pt,g in meta.groupby('PtID'):
  num=int(pt[3:]);path=f'AZT1D 2025/CGM Records/Subject {num}/Subject {num}.csv';raw=pd.read_csv(io.BytesIO(z.read(path)))
  raw['time']=pd.to_datetime(raw.EventDateTime,format='%Y-%m-%d %H:%M:%S',errors='coerce');raw['value']=pd.to_numeric(raw.CGM,errors='coerce')
  raw=raw[raw.time.notna()&np.isfinite(raw.value)&(raw.value>0)]
  count=raw.groupby('time').value.nunique();raw=raw[~raw.time.isin(count[count>1].index)].sort_values('time').drop_duplicates('time')
  times=raw.time.to_numpy(dtype='datetime64[ns]');v=raw.value.to_numpy();indices=g.index.to_numpy()
  for offset in list(range(-120,1,15))+[30,60,90]:
   boundary=g.timestamp.to_numpy(dtype='datetime64[ns]')+np.timedelta64(offset,'m');idx=np.searchsorted(times,boundary,side='right')-1;assert (idx>=0).all()
   age=(boundary-times[idx])/np.timedelta64(1,'m');assert (age>=0).all() and (age<15).all()
   expected=arr['X'][indices,(offset+120)//15,0] if offset<=0 else arr['y'][indices,[30,60,90].index(offset)]
   np.testing.assert_array_equal(v[idx].astype('float32'),expected);rows.append(dict(PtID=pt,offset_minutes=offset,n=len(g),max_age_minutes=float(age.max()),max_float32_rounding_mgdl=float(np.abs(v[idx]-expected).max())))
out=ROOT/'results/revision_v3/external';pd.DataFrame(rows).to_csv(out/'raw_boundary_verification.csv',index=False)
(out/'raw_boundary_verification.json').write_text(json.dumps({'status':'passed','origins':len(meta),'boundaries_checked':len(meta)*12,'method':'independent raw timestamp searchsorted; every input and target uses last raw observation within its own right-closed 15-minute bin','script_sha256':sha256(Path(__file__))},indent=2))
print('Verified',len(meta)*12,'raw observed boundaries')
