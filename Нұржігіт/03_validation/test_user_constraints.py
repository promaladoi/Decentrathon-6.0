from user_constraints import evaluate


def inputs():
    return ({'summary':{'base':{'eval_loss':2.05},'solution':{'eval_loss':2.049,'wall_s':29}},
             'runs':{'solution':[{'gpu_peak_mib':7151},{'gpu_peak_mib':7151}]}},
            {'checks':[{'ok':True} for _ in range(10)]})


def test_all_requirements_apply():
    r,c=inputs()
    assert evaluate(r,c)['all_requested_constraints_met']


def test_median_memory_cannot_hide_one_over_limit_run():
    r,c=inputs(); r['runs']['solution'][1]['gpu_peak_mib']=7153
    p=evaluate(r,c)
    assert not p['every_run_within_memory_limit'] and not p['all_requested_constraints_met']


def test_quality_target_and_original_ten_are_independent_gates():
    for change in ('quality','time','checks'):
        r,c=inputs()
        if change=='quality': r['summary']['solution']['eval_loss']=2.05001
        elif change=='time': r['summary']['solution']['wall_s']=30.01
        else: c['checks'][0]['ok']=False
        assert not evaluate(r,c)['all_requested_constraints_met']
