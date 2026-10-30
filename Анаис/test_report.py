import copy
import unittest
from benchmark_report import compare, loss_analysis


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.result = {'runs': {s: [dict(exit_code=0, wall_s=t, gpu_peak_mib=m)]
                               for s, t, m in [('base', 100, 1000), ('solution', 80, 700)]},
                       'evals': {s: dict(exit_code=0, mean_loss=2) for s in ('base', 'solution')}}
        self.checks = dict(result_sha256='abc', checks=[dict(ok=True) for _ in range(10)])

    def test_ratios_and_hash(self):
        _, m = compare(self.result, self.checks, 'abc')
        self.assertEqual(m['speedup'], 1.25)
        self.assertAlmostEqual(m['memory_reduction_pct'], 30)
        self.assertTrue(m['official_checks_passed'])
        self.assertFalse(compare(self.result, self.checks, 'stale')[1]['official_checks_passed'])

    def test_missing_failed_and_nonfinite(self):
        for value in [None, False, float('nan'), float('inf'), 0, -1]:
            result = copy.deepcopy(self.result)
            result['runs']['solution'][0]['wall_s'] = value
            with self.assertRaises(ValueError):
                compare(result, {}, '')
        self.result['evals']['solution']['exit_code'] = 1
        with self.assertRaises(ValueError):
            compare(self.result, {}, '')

    def test_median_and_max(self):
        self.result['runs']['base'].append(dict(exit_code=0, wall_s=120, gpu_peak_mib=1200))
        rows, _ = compare(self.result, {}, '')
        self.assertEqual(rows[0]['wall_s'], 110)
        self.assertEqual(rows[0]['gpu_peak_mib'], 1200)

    def test_curve_uses_mean_relative_error(self):
        self.result['runs']['base'][0]['log'] = {'loss': [2, 1]}
        self.result['runs']['solution'][0]['log'] = {'loss': [2.02, 1.03]}
        curve = loss_analysis(self.result)[1]
        self.assertAlmostEqual(curve['first_gap_pct'], 1)
        self.assertAlmostEqual(curve['mean_gap_pct'], 2)
        self.result['runs']['solution'][0]['log']['loss'] = [2]
        self.assertIsNone(loss_analysis(self.result)[1]['mean_gap_pct'])


if __name__ == '__main__':
    unittest.main()
