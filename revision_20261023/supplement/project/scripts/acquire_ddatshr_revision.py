"""Acquire only required public source files at a pinned Git commit; verify Git/LFS hashes."""
import hashlib,json,re
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
REPO='ictinnovaties-zorg/dataset-diabetes-adolescents-time-series-with-heart-rate'
COMMIT='36379d9397d05bb00ac20cf91a5a2b648fecfafe'
FILES={'Abbott.csv':'369d412f46e3cc20ddb63d3c7205611baec6b214',
       'Medtronic.csv':'85b89fd084c3fe15dc26c9aed9a55b4c4d858689',
       'population.csv':'c5a75a06742c4ee55ca66e454d8e241b08c19474',
       'Fitbit/Fitbit-heart-rate.csv':'1c5174076388145fa0825a2780042aa78d524cd6'}


def main():
    root=ROOT/'data/raw/ddatshr';root.mkdir(parents=True,exist_ok=True);records=[]
    for relative,blob in FILES.items():
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        upstream=f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/data-csv/{relative}'
        pointer=None
        if relative.startswith('Fitbit/'):
            r=requests.get(upstream,timeout=60);r.raise_for_status();pointer=r.text
            assert hashlib.sha1(b'blob '+str(len(r.content)).encode()+b'\0'+r.content).hexdigest()==blob
            expected=re.search(r'oid sha256:([0-9a-f]+)',pointer)[1]
        if not path.exists():
            url=f'https://media.githubusercontent.com/media/{REPO}/{COMMIT}/data-csv/{relative}' if pointer else upstream
            r=requests.get(url,stream=True,timeout=120);r.raise_for_status()
            with path.with_suffix('.part').open('wb') as f:
                for chunk in r.iter_content(8*1024*1024):f.write(chunk)
            path.with_suffix('.part').rename(path)
        sha=hashlib.sha256();git=hashlib.sha1(b'blob '+str(path.stat().st_size).encode()+b'\0')
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):sha.update(chunk);git.update(chunk)
        if pointer:assert sha.hexdigest()==expected,relative+' LFS mismatch'
        else:assert git.hexdigest()==blob,relative+' Git blob mismatch'
        records.append({'file':relative,'sha256':sha.hexdigest(),'git_blob':blob,'bytes':path.stat().st_size})
    r=requests.get(f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/README.md',timeout=60);r.raise_for_status()
    (root/'upstream_README.md').write_text(r.text)
    out=ROOT/'results/revision_v2/provenance';out.mkdir(parents=True,exist_ok=True)
    (out/'ddatshr_source_verification.json').write_text(json.dumps({'repository':REPO,'commit':COMMIT,'files':records},indent=2))
    print('All original DDATSHR files match pinned upstream Git/LFS objects.')

if __name__=='__main__':main()
