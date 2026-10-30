"""Separate CPU-only native parity proof; never a scored training result."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def child():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    sys.modules['unsloth'] = None
    sys.modules['unsloth_zoo'] = None
    import transformers
    from solution.metadata_scan import patch_metadata_scan
    patch_metadata_scan()
    from solution import deferred_metadata as integration
    from solution import package_metadata_snapshot as helper
    assert 'soup_cli.trainer.sft' not in sys.modules
    assert 'trl.import_utils' not in sys.modules
    assert integration.install()
    assert 'soup_cli.trainer.sft' not in sys.modules, 'integration must not import Soup early'
    # Exercise coexistence with the active v41 train-command import hook.
    sys.argv = ['soup', 'train', '--config', 'unused_cpu_prefix_probe.yaml', '--yes']
    from solution.lazy_train_cli import patch_lazy_train_cli
    assert patch_lazy_train_cli()
    import soup_cli.cli as cli
    assert cli.__solution_lazy_cli_info__['original_function_definitions_preserved'] >= 10
    assert 'soup_cli.trainer.sft' not in sys.modules
    assert 'trl.import_utils' not in sys.modules
    import soup_cli.trainer.sft as sft
    wrapped = sft.SFTTrainerWrapper.setup
    assert getattr(wrapped, helper._SETUP_MARKER, False)
    native = wrapped.__wrapped__
    original = helper.metadata.packages_distributions
    scans, snapshots, setups = [], [], []
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code is original.__code__:
            scans.append(frame.f_back.f_globals.get('__name__'))
        if event == 'call' and frame.f_code is native.__code__:
            setups.append('native')
        if (event == 'return' and frame.f_globals is vars(helper)
                and frame.f_code.co_name == 'packages_distributions'
                and frame.f_back is not None
                and frame.f_back.f_globals.get('__name__') == 'trl.import_utils'):
            snapshots.append(arg)
    class StopBeforeSeedAndModel(Exception):
        pass
    class Probe:
        @property
        def config(self):
            raise StopBeforeSeedAndModel()
    previous = sys.getprofile()
    stopped = False
    try:
        sys.setprofile(profile)
        try:
            wrapped(Probe(), {})
        except StopBeforeSeedAndModel:
            stopped = True
    finally:
        sys.setprofile(previous)
    assert stopped and setups == ['native']
    assert scans == [], scans
    assert len(snapshots) == 1
    assert helper.metadata.packages_distributions is original
    actual = sys.modules['trl.import_utils'].PACKAGE_DISTRIBUTION_MAPPING
    assert snapshots[0] is actual
    source = sys.modules['transformers.utils.import_utils'].PACKAGE_DISTRIBUTION_MAPPING
    assert type(actual) is dict and actual is not source
    assert all(actual[key] is not names for key, names in source.items())
    assert actual == source == original()
    torch = sys.modules.get('torch')
    assert torch is None or not torch.cuda.is_initialized()
    print('STARTUP_METADATA_NATIVE_PARITY', json.dumps(dict(
        native_mapping_equal=True, independent_dict_and_lists=True,
        active_lazy_cli_coexists=True,
        real_deferred_hook_hit=True, native_setup_calls=len(setups),
        actual_snapshot_returns_to_trl=len(snapshots), native_scans_during_setup=len(scans),
        original_function_restored=True, stopped_before_seed_and_model=True,
        cuda_initialized=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.child:
        child()
    else:
        # Independent proof process; no model, GPU allocations or cross-run compiler cache.
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child'],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True, timeout=120)


if __name__ == '__main__':
    main()
