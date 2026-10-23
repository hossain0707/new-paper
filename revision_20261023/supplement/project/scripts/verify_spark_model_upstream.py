"""Compare the complete local weight manifest to public pinned Hub metadata."""
import hashlib,json,re
from pathlib import Path
import requests
root=Path('/home/mkd-s006/data/model/qwen38-flash-next-fp4-local/checkpoint')
out=Path('/home/mkd-s006/glucosense_revision_20261022_2032/audit/model')
download=json.loads((root/'.keural-download.json').read_text())
repo=download['model'];revision=download['revision']
assert re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo)
assert re.fullmatch(r'[0-9a-f]{40}',revision)
url=f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
r=requests.get(url,timeout=60)
(out/'upstream_http_status.json').write_text(json.dumps({'url':url,'status':r.status_code},indent=2))
r.raise_for_status();remote=r.json()
(out/'upstream_model_metadata.json').write_text(json.dumps(remote,indent=2))
assert remote['sha']==revision
siblings={p['rfilename']:p for p in remote['siblings']}
manifest=json.loads((out/'weight_manifest.json').read_text());results=[]
for item in manifest['weight_files']:
    name=item['relative_path'];upstream=siblings.get(name,{})
    lfs=upstream.get('lfs') or {};expected=lfs.get('sha256') or lfs.get('oid')
    results.append({'file':name,'local_sha256':item['sha256'],'upstream_lfs_sha256':expected,
                    'matches':bool(expected and item['sha256']==expected)})
for name,digest in manifest['metadata_sha256'].items():
    upstream=siblings.get(name,{});data=(root/name).read_bytes();lfs=upstream.get('lfs') or {}
    expected=lfs.get('sha256') or lfs.get('oid')
    actual=hashlib.sha256(data).hexdigest() if expected else hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if not expected:expected=upstream.get('blobId')
    results.append({'file':name,'local_git_or_lfs_hash':actual,'upstream_git_or_lfs_hash':expected,'matches':bool(expected and actual==expected)})
report={'repository':repo,'revision':revision,'files_checked':len(results),'all_match':all(x['matches'] for x in results),'checks':results,
        'weight_manifest_sha256':hashlib.sha256((out/'weight_manifest.json').read_bytes()).hexdigest()}
(out/'upstream_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
