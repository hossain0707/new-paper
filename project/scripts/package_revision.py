"""Build an explicit allowlisted supplement, without source patient records or correspondence."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v2'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    assert json.loads((OUT/'artifact_verification.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'results/revision_v3/artifact_verification.json').read_text())['status']=='passed'
    paths=set()
    for pattern in ['src/revision/*.py','src/models/deep_models.py','src/evaluation/metrics.py','src/llm/safety_filters.py',
                    'tests/*.py','scripts/*revision*.py','scripts/build_mdpi_docx.py',
                    'requirements-revision.txt','requirements-vllm-lock.txt','requirements-chronos.txt',
                    'README.md','REPRODUCIBILITY.md','protocol_v2.md','protocol_v3.md','LITERATURE_AUDIT.md',
                    'PRIOR_PUBLICATION_DISCLOSURE.md','SUBMISSION_READINESS.md',
                    'manuscript/*REVISED_EVIDENCE.md','manuscript/revision_template.md',
                    'results/revision_v2/tables/*','results/revision_v2/figures/*',
                    'results/revision_v2/provenance/*.json','results/revision_v2/*.json',
                    'results/revision_v2/*.md','results/revision_v2/*.log',
                    'results/revision_v2/llm/*.csv','results/revision_v2/llm/*freeze.json',
                    'results/revision_v2/llm/guardrail_challenge.json',
                    'results/revision_v2/fits/**/complete.json',
                    'results/revision_v2/sensitivity/**/complete.json',
                    'results/revision_v3/*.json','results/revision_v3/tables/*.csv',
                    'results/revision_v3/figures/*','results/revision_v3/provenance/*',
                    'results/revision_v3/external/*.json','results/revision_v3/external/*.csv',
                    'results/revision_v3/chronos/*.json','results/revision_v3/latency/*.json',
                    'results/revision_v3/latency/*.csv','results/revision_v3/**/fits/**/complete.json']:
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    paths={p for p in paths if p.name!='package_manifest.json'}
    def archive_name(p):
        relative=p.relative_to(ROOT)
        # Audit copies must not masquerade as completed caches without their weights.
        if p.name=='complete.json' and 'fits' in relative.parts:
            return f'results/{relative.parts[1]}/provenance/fit_manifests/'+'/'.join(relative.parts[2:])
        return str(relative)
    files={archive_name(p):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(paths)}
    manifest={'files':files,'exclusions':'No raw data, prepared arrays, per-origin predictions, real generation cases/responses, weights or private professor correspondence. Acquire and regenerate using REPRODUCIBILITY.md. Local originals are retained. Completion manifests are relocated under provenance/fit_manifests to avoid creating incomplete resumable caches in a fresh extraction.',
              'verification':'SHA-256 applies to the exact included bytes; this archive is prepared locally, not a claim of public release.'}
    (OUT/'package_manifest.json').write_text(json.dumps(manifest,indent=2))
    destination=ROOT.parent/'revision_evidence_supplement.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(paths):archive.write(p,archive_name(p))
        archive.write(OUT/'package_manifest.json','MANIFEST.json')
        archive.writestr('READ_ME_FIRST.txt','Corrected retrospective research supplement. Start with README.md and REPRODUCIBILITY.md. Author declarations and journal approval remain pending. No source participant records are re-hosted. Model predictions and weights can be regenerated from the pinned sources.\n')
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name,record in files.items():assert hashlib.sha256(archive.read(name)).hexdigest()==record['sha256']
    destination.with_suffix('.zip.sha256').write_text(f'{digest(destination)}  {destination.name}\n')
    print(f'Verified {len(files)} files; {destination.stat().st_size:,} bytes: {destination}')

if __name__=='__main__':main()
