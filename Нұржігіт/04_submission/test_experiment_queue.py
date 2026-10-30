import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
import zipfile
import pytest

spec = importlib.util.spec_from_file_location('queue_runner', Path(__file__).with_name('coordinate_experiments.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def valid():
    return {'schema': 1, 'jobs': [{'version': 'v17', 'commit': 'a'*40, 'nf4_analysis': True}]}


def test_valid_queue_round_trip():
    q = valid()
    assert runner.parse_queue(json.dumps(q)) == q['jobs']


@pytest.mark.parametrize('field,value', [('version','../v17'),('commit','main'),('nf4_analysis','yes')])
def test_rejects_unsafe_job_fields(field,value):
    q=valid(); q['jobs'][0][field]=value
    with pytest.raises(ValueError): runner.parse_queue(json.dumps(q))


def test_rejects_commands_and_duplicate_versions():
    q=valid(); q['jobs'][0]['command']='anything'
    with pytest.raises(ValueError): runner.parse_queue(json.dumps(q))
    q=valid(); q['jobs']*=2
    with pytest.raises(ValueError): runner.parse_queue(json.dumps(q))


def make_archive(tmp_path, status='accepted', count=10, all_ok=True, corrupt=False):
    p=tmp_path/'artifact.zip'; result=b'{"summary":{"synthetic_test_fixture":true}}'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('experiment_outcome.json',json.dumps({'code_commit':'a'*40,'status':status}))
        z.writestr('result.json',result)
        z.writestr('checks.json',json.dumps({'result_sha256':'wrong' if corrupt else hashlib.sha256(result).hexdigest(),
                                             'checks':[{'ok':all_ok} for _ in range(count)]}))
    return p


@pytest.mark.parametrize('status,count,ok,accepted', [('accepted',10,True,True),('failed',10,True,False),
                                                   ('accepted',9,True,False),('accepted',10,False,False)])
def test_only_complete_matching_reports_are_accepted(tmp_path,status,count,ok,accepted):
    result=runner.archived_outcome(make_archive(tmp_path,status,count,ok),'a'*40)
    assert result['accepted'] is accepted


def test_rejects_wrong_source_or_hash(tmp_path):
    with pytest.raises(ValueError): runner.archived_outcome(make_archive(tmp_path),'b'*40)
    with pytest.raises(ValueError): runner.archived_outcome(make_archive(tmp_path,corrupt=True),'a'*40)


def test_pinned_queue_survives_new_branch_manifest(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True).strip()
    git('init', '-q')
    git('config', 'user.email', 'queue-test@example.invalid')
    git('config', 'user.name', 'Queue Test')
    manifest = tmp_path/'04_submission'/'experiment_queue.json'
    manifest.parent.mkdir()
    old = valid()
    manifest.write_text(json.dumps(old))
    git('add', '.')
    git('commit', '-qm', 'old queue')
    pinned = git('rev-parse', 'HEAD')
    new = valid()
    new['jobs'][0]['version'] = 'v44'
    manifest.write_text(json.dumps(new))
    git('add', '.')
    git('commit', '-qm', 'new queue')
    git('fetch', '-q', '.', 'HEAD')
    assert runner.read_queue(tmp_path, pinned) == old['jobs']
    assert runner.read_queue(tmp_path) == new['jobs']


@pytest.mark.parametrize('revision', ['main', 'HEAD:other', 'a'*39])
def test_queue_pin_rejects_non_sha_before_git(tmp_path, revision):
    with pytest.raises(ValueError, match='complete SHA'):
        runner.read_queue(tmp_path, revision)
