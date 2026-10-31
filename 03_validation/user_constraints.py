"""Additional user limits, evaluated after the original organizer checks."""
import argparse
import hashlib
import json
from pathlib import Path


def evaluate(result, checks, target_seconds=25):
    solutions = result['runs']['solution']
    base, solution = result['summary']['base'], result['summary']['solution']
    original_ten = len(checks['checks']) == 10 and all(c['ok'] for c in checks['checks'])
    peak = max(r['gpu_peak_mib'] for r in solutions)
    memory = peak <= 7151
    quality = solution['eval_loss'] <= base['eval_loss']
    target = solution['wall_s'] <= target_seconds
    return dict(original_ten_passed=original_ten, max_solution_peak_mib=peak,
                memory_limit_mib=7151, every_run_within_memory_limit=memory,
                reported_eval_loss_nonincreasing=quality,
                base_eval_loss=base['eval_loss'], solution_eval_loss=solution['eval_loss'],
                median_total_seconds=solution['wall_s'], target_total_seconds=target_seconds,
                target_met=target, eligible_within_memory_and_quality=original_ten and memory and quality,
                all_requested_constraints_met=original_ten and memory and quality and target,
                quality_note='Compares the reported rounded evaluation loss; original loss-curve checks remain required.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('result',type=Path)
    args=parser.parse_args()
    raw=args.result.read_bytes()
    check_raw=(args.result.parent/'checks.json').read_bytes()
    checks=json.loads(check_raw)
    assert hashlib.sha256(raw).hexdigest()==checks['result_sha256']
    report=evaluate(json.loads(raw),checks)
    report.update(result_sha256=checks['result_sha256'],solution_sha256=checks['solution_sha256'])
    dest=args.result.parent/'user_constraints.json'
    if dest.exists():
        raise FileExistsError('Preserve the prior constraints report')
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__': main()
