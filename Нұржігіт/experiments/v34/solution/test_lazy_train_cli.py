"""CPU tests of the real pinned CLI parser and guarded import transformation."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from solution.lazy_train_cli import CLI_SHA256, patch_lazy_train_cli, train_only_source


def reference():
    spec = importlib.util.find_spec('soup_cli.cli')
    return spec.loader.get_source('soup_cli.cli'), spec.origin


def test_original_functions_callback_and_train_registration_preserved():
    source, name = reference()
    prepared = train_only_source(source, name)
    assert prepared is not None
    _, info = prepared
    assert info['original_source_sha256'] == CLI_SHA256
    assert info['deferred_command_bindings'] == 84
    assert info['original_function_definitions_preserved'] >= 10
    assert info['train_registrations'] == 1


def test_source_changes_decline_the_optimization():
    source, name = reference()
    assert train_only_source(source+'\n# changed upstream\n', name) is None


@pytest.mark.parametrize('argv', [['soup', '--help'], ['soup', 'chat'], ['pytest']])
def test_other_commands_leave_import_finders_unchanged(monkeypatch, argv):
    monkeypatch.setattr(sys, 'argv', argv)
    before = list(sys.meta_path)
    assert not patch_lazy_train_cli()
    assert sys.meta_path == before


def invoke(args, patched):
    prefix = ('from solution.lazy_train_cli import patch_lazy_train_cli;'
              'assert patch_lazy_train_cli();') if patched else ''
    code = ('import sys;sys.argv='+repr(['soup', 'train', *args])+';'+prefix+
            'from soup_cli.cli import run;run()')
    env = dict(os.environ, NO_COLOR='1', TERM='dumb', COLUMNS='160',
               SOUP_NO_AUDIT_LOG='1', SOUP_NO_TELEMETRY='1')
    done = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                           env=env, text=True, encoding='utf-8', stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=90)
    lines = [line for line in done.stdout.splitlines()
             if not line.startswith('solution: train CLI deferred ')]
    return done.returncode, '\n'.join(lines)


def test_real_train_help_is_identical():
    expected = invoke(['--help'], False)
    actual = invoke(['--help'], True)
    assert actual[0] == expected[0] == 0, actual
    assert actual[1] == expected[1]


def test_real_missing_config_keeps_parser_and_error_behavior():
    args = ['--config', 'this_config_does_not_exist_for_cli_contract_test.yaml', '--yes']
    expected, actual = invoke(args, False), invoke(args, True)
    assert actual[0] == expected[0] != 0
    assert actual[1] == expected[1]
