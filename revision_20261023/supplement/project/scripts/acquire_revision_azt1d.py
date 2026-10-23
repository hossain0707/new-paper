"""Acquire the openly licensed, pinned AZT1D v1 archive; no registration or messages."""
import hashlib,json,os
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
URL='https://data.mendeley.com/public-files/datasets/gk9m674wcx/files/b02a20be-27c4-4dd0-8bb5-9171c66262fb/file_downloaded'
SHA='2c6b4183fdc729c708c1b3230ab1988d2c9006dd35729537f2de07f649f0d113'
SIZE=776131685
p=ROOT/'data/raw/azt1d_v1/AZT1D_2025.zip';p.parent.mkdir(parents=True,exist_ok=True)
def verify(path):
    assert path.stat().st_size==SIZE,'Archive size differs from pinned release'
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==SHA,'Archive checksum mismatch'
if p.exists():verify(p)
else:
    tmp=p.with_suffix('.download')
    with requests.get(URL,stream=True,timeout=(30,120)) as response:
        response.raise_for_status()
        with tmp.open('wb') as f:
            for block in response.iter_content(1024*1024):f.write(block)
    verify(tmp);os.replace(tmp,p)
out=ROOT/'results/revision_v3/provenance';out.mkdir(parents=True,exist_ok=True)
(out/'azt1d_acquisition.json').write_text(json.dumps({'doi':'10.17632/gk9m674wcx.1','version':1,'license':'CC BY 4.0','provider_page':'https://data.mendeley.com/datasets/gk9m674wcx/1','download_url':URL,'sha256':SHA,'bytes':SIZE,'status':'verified'},indent=2))
print('Verified pinned AZT1D v1 archive')
