"""Serialize the complete Qwen tokenizer state with C pickle during hashing.

Dataset cache fingerprints are internal identifiers, not training inputs. This
keeps all tokenizer state and the original datasets normalization/restoration of
runtime padding/truncation, but avoids walking a 151k-token vocabulary in Python
dill. It changes the cache-key representation; it never reuses another run's
tokenized rows or skips tokenization/validation.
"""
import ast
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import pickle
import sys

HANDLER_SHA256='8e3ce3ee43fc57b1b02f203238679087d5405b3f99b16889f33c1580e5bad755'
MODULE='datasets.utils._dill'


def canonical(value, memo):
    if type(value) not in (dict,list,tuple):
        return value
    if id(value) in memo:
        return memo[id(value)]
    if type(value) is dict:
        out={}
        memo[id(value)]=out
        out.update((k,canonical(value[k],memo)) for k in sorted(value))
        return out
    if type(value) is list:
        out=[]
        memo[id(value)]=out
        out.extend(canonical(v,memo) for v in value)
        return out
    out=tuple(canonical(v,memo) for v in value)
    memo[id(value)]=out
    return out


def payload_reduce(pickler,obj,state):
    cls=type(obj)
    if (cls.__module__!='transformers.models.qwen2.tokenization_qwen2' or
            cls.__name__!='Qwen2Tokenizer' or cls.__new__ is not object.__new__):
        return pickler.save_reduce(cls,(),state=state,obj=obj)
    try:
        clone=object.__new__(cls)
        clone.__dict__.update(canonical(state,{}))
        payload=pickle.dumps(clone,protocol=4)
    except (TypeError,ValueError,RecursionError,pickle.PicklingError):
        return pickler.save_reduce(cls,(),state=state,obj=obj)
    return pickler.save_reduce(pickle.loads,(payload,),obj=obj)


def install(module):
    original=module._save_transformersPreTrainedTokenizerBase
    if getattr(original,'_solution_c_pickle',False):
        return True
    source=inspect.getsource(original).strip()
    if hashlib.sha256(source.encode()).hexdigest()!=HANDLER_SHA256:
        return False
    tree=ast.parse(source)
    count=0
    class Rewrite(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal count
            if (isinstance(node.func,ast.Attribute) and
                    isinstance(node.func.value,ast.Name) and node.func.value.id=='pickler' and
                    node.func.attr=='save_reduce'):
                count+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='_solution_payload_reduce',ctx=ast.Load()),
                    args=[ast.Name(id=n,ctx=ast.Load()) for n in ('pickler','obj','state')],keywords=[]),node)
            return self.generic_visit(node)
    tree=Rewrite().visit(tree)
    if count!=1:
        return False
    namespace=dict(vars(module),_solution_payload_reduce=payload_reduce)
    exec(compile(ast.fix_missing_locations(tree),original.__code__.co_filename,'exec'),namespace)
    prepared=namespace[original.__name__]
    prepared._solution_c_pickle=True
    module._save_transformersPreTrainedTokenizerBase=prepared
    for kind,handler in list(module.Pickler.dispatch.items()):
        if handler is original:
            module.Pickler.dispatch[kind]=prepared
    return True


class Loader(importlib.abc.Loader):
    def __init__(self,original): self.original=original
    def create_module(self,spec): return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        install(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=MODULE:
            return None
        if self in sys.meta_path: sys.meta_path.remove(self)
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is not None and spec.loader is not None:
            spec.loader=Loader(spec.loader)
        return spec


def patch_tokenizer_fingerprint():
    if MODULE in sys.modules:
        return install(sys.modules[MODULE])
    if not any(isinstance(f,Finder) for f in sys.meta_path):
        sys.meta_path.insert(0,Finder())
    return True
