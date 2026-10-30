"""Build an auditable report from the unmodified official harness artifacts."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def loss_analysis(result):
    """Compare every run to base_1 using the official relative-gap definition."""
    base_runs = result.get('runs', {}).get('base', [])
    reference = base_runs[0].get('log', {}).get('loss', []) if base_runs else []
    rows = []
    for side in ('base', 'solution'):
        for run in result.get('runs', {}).get(side, []):
            loss = run.get('log', {}).get('loss', [])
            valid = (isinstance(loss, list) and isinstance(reference, list) and bool(loss)
                     and len(loss) == len(reference) and all(positive(x) for x in loss + reference))
            rows.append(dict(run=run.get('run', side), steps=len(loss) if isinstance(loss, list) else 0,
                             first_loss=loss[0] if valid else None,
                             mean_loss=statistics.mean(loss) if valid else None,
                             first_gap_pct=100*abs(loss[0]-reference[0])/max(abs(reference[0]), 1e-12) if valid else None,
                             mean_gap_pct=100*statistics.mean(abs(a-b)/max(abs(b), 1e-12)
                                                             for a, b in zip(loss, reference)) if valid else None))
    return rows


def compare(result, checks, digest):
    rows = []
    for side in ('base', 'solution'):
        runs = result.get('runs', {}).get(side, [])
        if not runs or any(r.get('exit_code') != 0 for r in runs):
            raise ValueError(f'{side}: missing or failed training runs')
        for r in runs:
            if not all(positive(r.get(k)) for k in ('wall_s', 'gpu_peak_mib')):
                raise ValueError(f'{side}: invalid time or memory')
        evaluation = result.get('evals', {}).get(side, {})
        loss = evaluation.get('mean_loss')
        if evaluation.get('exit_code') != 0 or not positive(loss):
            raise ValueError(f'{side}: missing or failed evaluation')
        rows.append(dict(side=side, runs=len(runs), wall_s=statistics.median(r['wall_s'] for r in runs),
                         gpu_peak_mib=max(r['gpu_peak_mib'] for r in runs), eval_loss=loss))
    b, s = rows
    recorded = checks.get('checks', [])
    verified = (checks.get('result_sha256') == digest and isinstance(recorded, list)
                and len(recorded) == 10 and all(isinstance(c, dict) and c.get('ok') is True for c in recorded))
    return rows, dict(speedup=b['wall_s']/s['wall_s'],
                      time_reduction_pct=100*(1-s['wall_s']/b['wall_s']),
                      memory_reduction_pct=100*(1-s['gpu_peak_mib']/b['gpu_peak_mib']),
                      quality_delta_pct=100*(s['eval_loss']/b['eval_loss']-1),
                      official_checks_passed=verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, default=Path('runs/result.json'))
    parser.add_argument('--checks', type=Path)
    parser.add_argument('--out', type=Path, default=Path('anais/reports/latest'))
    args = parser.parse_args()
    raw = args.result.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    checks_path = args.checks or args.result.with_name('checks.json')
    checks = json.loads(checks_path.read_text(encoding='utf-8')) if checks_path.exists() else {}
    result = json.loads(raw)
    rows, metrics = compare(result, checks, digest)
    curves = loss_analysis(result)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if curves:
        with (args.out/'loss_analysis.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(curves[0]))
            writer.writeheader()
            writer.writerows(curves)
    status = 'PASS: 10 official checks, matching result SHA256' if metrics['official_checks_passed'] else 'NOT VERIFIED: official checks failed, absent or stale'
    text = ['# Benchmark report', '', status, '', f'Result SHA256: `{digest}`', '',
            '| Run | Count | Time, s (median) | Peak MiB (max) | Eval loss |',
            '|---|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['side']} | {row['runs']} | {row['wall_s']:.2f} | {row['gpu_peak_mib']} | {row['eval_loss']:.6f} |")
    text += ['', f"Speedup: {metrics['speedup']:.4f}x; time saved: {metrics['time_reduction_pct']:.3f}%.",
             f"Memory saved: {metrics['memory_reduction_pct']:.3f}%; eval loss change: {metrics['quality_delta_pct']:+.4f}%.",
             '', 'Negative time savings mean a slowdown. Negative loss change means lower loss.',
             'These public-data measurements do not establish hidden-data performance or final jury admission.',
             'Checks authenticate result bytes, not the current solution files; rerun checks after code changes.', '', '## Official checks']
    for check in checks.get('checks', []):
        text.append(f"- {'PASS' if check.get('ok') is True else 'FAIL'} {check.get('check')}: {check.get('detail')}")
    text += ['', '## Training loss', '',
             '| Run | Logged steps | First loss | Mean loss | First gap % | Mean relative gap % |',
             '|---|---:|---:|---:|---:|---:|']
    for curve in curves:
        values = [str(curve['run']), str(curve['steps'])]
        values += ['missing' if curve[k] is None else f'{curve[k]:.6f}'
                   for k in ('first_loss', 'mean_loss', 'first_gap_pct', 'mean_gap_pct')]
        text.append('| ' + ' | '.join(values) + ' |')
    text += ['', 'Gaps use base_1 as reference. Limits: first step 1%, mean relative gap 3%.',
             'Training loss and held-out evaluation loss measure different data and must not be conflated.',
             '', '## Individual timings', '', '| Run | Wall s | Peak MiB | Exit code |', '|---|---:|---:|---:|']
    for side in ('base', 'solution'):
        for run in result['runs'][side]:
            text.append(f"| {run.get('run', side)} | {run['wall_s']} | {run['gpu_peak_mib']} | {run['exit_code']} |")
    text += ['', 'Two repeats describe variability but are insufficient for a statistical significance claim.',
             '', '## Environment', '```json', json.dumps(result.get('versions', {}), indent=2), '```']
    (args.out/'REPORT.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    (args.out/'metrics.json').write_text(json.dumps(dict(result_sha256=digest, **metrics), indent=2)+'\n', encoding='utf-8')
    print(status)
    print(args.out.resolve())


if __name__ == '__main__':
    main()
