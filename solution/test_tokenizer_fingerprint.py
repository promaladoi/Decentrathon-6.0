import copy
import pickle

import pytest

from .tokenizer_fingerprint import canonical, install, payload_reduce


@pytest.fixture
def tokenizer():
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    return Qwen2Tokenizer(vocab={'<|endoftext|>':0,'a':1,'b':2,'c':3},merges=[])


@pytest.fixture
def patched(monkeypatch):
    import datasets.utils._dill as d
    # Restore both the global handler and its type dispatch table after each test.
    monkeypatch.setattr(d,'_save_transformersPreTrainedTokenizerBase',d._save_transformersPreTrainedTokenizerBase)
    monkeypatch.setattr(d.Pickler,'dispatch',copy.copy(d.Pickler.dispatch))
    assert install(d)
    assert install(d)
    return d


def test_roundtrip_retains_all_state_and_outputs(tokenizer,patched):
    tokenizer.chat_template="{% for m in messages %}{{ m['content'] }}{% endfor %}"
    expected=tokenizer(['abc','ab'],padding=True,truncation=True,max_length=5)
    clone=pickle.loads(patched.dumps(tokenizer))
    assert clone(['abc','ab'],padding=True,truncation=True,max_length=5)==expected
    assert clone.get_vocab()==tokenizer.get_vocab()
    assert clone.chat_template==tokenizer.chat_template
    assert clone.padding_side==tokenizer.padding_side


def test_hash_stable_across_calls_runtime_settings_and_dict_order(tokenizer,patched):
    first=patched.dumps(tokenizer)
    tokenizer(['abc','ab'],padding=True,truncation=True,max_length=3)
    padding=copy.deepcopy(tokenizer._tokenizer.padding)
    truncation=copy.deepcopy(tokenizer._tokenizer.truncation)
    assert patched.dumps(tokenizer)==first
    assert tokenizer._tokenizer.padding==padding
    assert tokenizer._tokenizer.truncation==truncation
    clone=copy.deepcopy(tokenizer)
    clone._vocab=dict(reversed(list(clone._vocab.items())))
    assert patched.dumps(clone)==first


@pytest.mark.parametrize('field,value',[
    ('chat_template','different template'),('padding_side','left'),
    ('truncation_side','left'),('model_max_length',42),
])
def test_relevant_state_changes_fingerprint(tokenizer,patched,field,value):
    before=patched.dumps(tokenizer)
    setattr(tokenizer,field,value)
    assert patched.dumps(tokenizer)!=before


def test_vocab_addition_changes_fingerprint(tokenizer,patched):
    before=patched.dumps(tokenizer)
    tokenizer.add_tokens(['new-token'])
    assert patched.dumps(tokenizer)!=before


def test_other_objects_fall_back_without_loss():
    calls=[]
    class Record:
        def save_reduce(self,*a,**kw): calls.append((a,kw))
    obj=object()
    payload_reduce(Record(),obj,{'a':1})
    assert calls[0][0]==(object,()) and calls[0][1]=={'state':{'a':1},'obj':obj}


def test_canonical_preserves_container_aliases_and_cycles():
    child=['x']; source={'b':child,'a':child}; child.append(source)
    out=canonical(source,{})
    assert list(out)==['a','b'] and out['a'] is out['b']
    assert out['a'][1] is out


def _recursive_reference(value, memo):
    if type(value) not in (dict, list, tuple):
        return value
    if id(value) in memo:
        return memo[id(value)]
    if type(value) is dict:
        out = {}; memo[id(value)] = out
        out.update((k, _recursive_reference(value[k], memo)) for k in sorted(value))
    elif type(value) is list:
        out = []; memo[id(value)] = out
        out.extend(_recursive_reference(v, memo) for v in value)
    else:
        out = tuple(_recursive_reference(v, memo) for v in value)
        memo[id(value)] = out
    return out


@pytest.mark.parametrize('tuple_keys', [False, True])
def test_large_immutable_map_retains_exact_reference_pickle_and_aliases(tuple_keys):
    scalar = [None, b'bytes', True, 2, 3.25, 'text']
    keys = [(i, i + 1) if tuple_keys else f'token-{i}' for i in range(4096)]
    mapping = {k: scalar[i % len(scalar)] for i, k in reversed(list(enumerate(keys)))}
    state = {'vocab': mapping, 'alias': mapping, 'other': ['nested', mapping]}
    expected = _recursive_reference(state, {})
    actual = canonical(state, {})
    assert pickle.dumps(actual, protocol=4) == pickle.dumps(expected, protocol=4)
    assert actual['vocab'] is actual['alias'] is actual['other'][1]
    assert list(mapping) == list(reversed(keys))


def test_large_map_with_container_value_keeps_recursive_cycle_fallback():
    state = {str(i): i for i in range(1024)}
    state['cycle'] = state
    actual = canonical(state, {})
    assert actual['cycle'] is actual
    assert pickle.dumps(actual, protocol=4) == pickle.dumps(_recursive_reference(state, {}), protocol=4)


def test_dataset_map_outputs_unchanged(tokenizer,patched):
    from datasets import Dataset
    rows=Dataset.from_list([{'text':'abc'},{'text':'ab'}])
    fn=lambda row: tokenizer(row['text'])
    mapped=rows.map(fn,remove_columns=['text'])
    assert mapped.to_dict()=={'input_ids':[[1,2,3],[1,2]],'attention_mask':[[1,1,1],[1,1]]}
