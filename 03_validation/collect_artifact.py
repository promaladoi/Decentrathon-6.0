"""Verify exact Git sources and original report hashes before collecting evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile


def collect(archive, commit, version, repo, outputs=None):
    if not re.fullmatch(r'[0-9a-f]{40}', commit) or not re.fullmatch(r'v[0-9]+[a-z0-9_]*', version):
        raise ValueError('exact source SHA and safe version required')
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit,
                                     '--', 'solution'], cwd=repo, text=True).splitlines()
    expected = {n for n in names if n.endswith('.py')}
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('archive corruption or duplicate names')
        actual = {n for n in z.namelist() if n.startswith('solution/') and n.endswith('.py')}
        if actual != expected:
            raise ValueError('source file set differs from the exact Git revision')
        digest = hashlib.sha256()
        for name in sorted(expected):
            raw = z.read(name)
            if raw != subprocess.check_output(['git', 'show', commit+':'+name], cwd=repo):
                raise ValueError('source differs: '+name)
            digest.update(name.split('/', 1)[1].encode())
            digest.update(raw)
        outcome = json.loads(z.read('experiment_outcome.json'))
        if outcome['code_commit'] != commit or outcome['version'] != version:
            raise ValueError('outcome identifies another source or experiment')
        accepted, passed, summary = False, None, None
        has_result, has_checks = 'result.json' in z.namelist(), 'checks.json' in z.namelist()
        if has_checks:
            if not has_result:
                raise ValueError('checks without result')
            result_raw = z.read('result.json')
            checks = json.loads(z.read('checks.json'))
            if (checks['result_sha256'] != hashlib.sha256(result_raw).hexdigest() or
                    checks['solution_sha256'] != digest.hexdigest()):
                raise ValueError('original result/source checksum mismatch')
            passed = sum(bool(c['ok']) for c in checks['checks'])
            accepted = outcome['status'] == 'accepted' and len(checks['checks']) == passed == 10
            summary = json.loads(result_raw)['summary']
        if outcome['status'] == 'accepted' and not accepted:
            raise ValueError('accepted outcome lacks ten verified checks')
        evidence = {n: z.read(n) for n in z.namelist()
                    if len(Path(n).parts) == 1 and n.endswith(('.json', '.log'))}
    report = {'version': version, 'source_commit': commit, 'source_files_verified': len(expected),
              'solution_sha256': digest.hexdigest(), 'status': outcome['status'],
              'checks_passed': passed, 'accepted': accepted, 'summary': summary,
              'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    dest = repo/'03_validation/reports'/('codex_'+version)
    dest.mkdir(parents=True, exist_ok=True)
    for name, raw in evidence.items():
        (dest/name).write_bytes(raw)
    (dest/'collection_proof.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    if accepted and outputs is not None:
        shutil.copyfile(archive, outputs/('codex_'+version+'.zip'))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('archive', type=Path)
    p.add_argument('--commit', required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--outputs', type=Path)
    a = p.parse_args()
    print(json.dumps(collect(a.archive, a.commit, a.version, Path(__file__).resolve().parents[1],
                             a.outputs), ensure_ascii=False))
