"""Download and checksum the public raw DiaData archive, preserving its licence metadata."""
import hashlib
import json
from pathlib import Path
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / 'data/raw/diadata';root.mkdir(parents=True, exist_ok=True)
    response = requests.get('https://zenodo.org/api/records/17285631', timeout=60)
    response.raise_for_status();record=response.json()
    (root/'record_17285631.json').write_text(json.dumps(record, indent=2))
    entry = next(x for x in record['files'] if x['key']=='15_min_sampling.zip')
    archive=root/'15_min_sampling_complete.zip'
    if not archive.exists():
        r=requests.get(entry['links']['self'], stream=True, timeout=120);r.raise_for_status()
        tmp=archive.with_suffix('.part')
        with tmp.open('wb') as f:
            for b in r.iter_content(8*1024*1024):f.write(b)
        tmp.rename(archive)
    md5=hashlib.md5()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):md5.update(b)
    assert entry['checksum']=='md5:'+md5.hexdigest(), 'Source checksum mismatch'
    def extract(z):
        for name in z.namelist():
            if ('raw' in name.lower() and name.endswith('.zip')) or ('SDBII_' in name and name.endswith('.csv')):
                path=root/Path(name).name
                path.write_bytes(z.read(name))
                if path.suffix=='.zip':
                    with zipfile.ZipFile(path) as child:
                        for member in child.namelist():
                            if 'SDBII_' in member:
                                dst=root/Path(member).name;dst.write_bytes(child.read(member))
                                if dst.suffix=='.zip':
                                    with zipfile.ZipFile(dst) as last:
                                        for m in last.namelist():
                                            if m.endswith('.csv'):(root/Path(m).name).write_bytes(last.read(m))
    with zipfile.ZipFile(archive) as z:extract(z)
    assert (root/'SDBII_15min_sampling_raw.csv').exists()
    print('Verified raw DiaData acquired; original records retained locally.')


if __name__=='__main__':main()
