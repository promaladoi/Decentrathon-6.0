"""CPU address/layout check of the ACTUAL paired kernel AST, not GPU codegen.

Executes that pure kernel body through a small NumPy pointer model. The oracle
uses unpackbits/repeat rather than the kernel's byte shifts and pointer math.
Dyadic data isolates layout/tail correctness from GPU FMA differences.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np

root=Path(__file__).resolve().parent.parent
repo=root
sha=subprocess.check_output(['git','rev-parse','2c2baf4'],cwd=repo,text=True).strip()
raw=subprocess.check_output(['git','show',sha+':solution/paired_nf4.py'],cwd=repo)


class Vector(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Vector)


def vector(value):
    return np.asarray(value).view(Vector)


class Pointer:
    def __init__(self, values, offset=0):
        self.values, self.offset = values, offset
        self.dtype=SimpleNamespace(element_ty=values.dtype)
    def __add__(self, offset):
        return Pointer(self.values, self.offset+offset)


class Language:
    int32, int64 = np.int32, np.int64
    pid=0
    def program_id(self, axis):
        assert axis==0
        return vector(np.int64(self.pid))
    def arange(self, start, end):
        return vector(np.arange(start,end,dtype=np.int64))
    def load(self, pointer, mask=True, other=0):
        indices, mask=np.broadcast_arrays(np.asarray(pointer.offset),np.asarray(mask))
        if np.any(mask & ((indices<0)|(indices>=pointer.values.size))):
            raise AssertionError('unmasked out-of-bounds read')
        safe=np.clip(indices,0,pointer.values.size-1).astype(np.int64)
        return vector(np.where(mask,pointer.values[safe],np.array(other,dtype=pointer.values.dtype)))
    def store(self, pointer, values, mask=True):
        indices, values, mask=np.broadcast_arrays(np.asarray(pointer.offset),np.asarray(values),np.asarray(mask))
        if np.any(mask & ((indices<0)|(indices>=pointer.values.size))):
            raise AssertionError('unmasked out-of-bounds write')
        pointer.values[indices[mask].astype(np.int64)]=values[mask]
    def join(self,a,b):
        return vector(np.stack((a,b),axis=-1))
    def reshape(self,a,shape):
        return vector(a.reshape(shape))


node=copy.deepcopy(next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='paired_nf4_kernel'))
node.decorator_list=[]
node.returns=None
for arg in node.args.args: arg.annotation=None
module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
tl=Language()
scope={'tl':tl}
exec(compile(module,'<actual paired_nf4_kernel body>','exec'),scope)
kernel=scope['paired_nf4_kernel']
lengths=[1,2,63,64,65,511,512,2047,2048,2049,4097,16385,32769]
cases=[]
for n in lengths:
    packed=((np.arange((n+1)//2,dtype=np.int64)*37+17)%256).astype(np.uint8)
    bits=np.unpackbits(packed).reshape(-1,2,4)
    nibbles=(bits*np.array([8,4,2,1],dtype=np.uint8)).sum(-1).reshape(-1)[:n]
    code=(np.arange(16,dtype=np.float32)-8)/8
    for qblock in (64,128):
        blocks=(n+qblock-1)//qblock
        for nested in (False,True):
            code2=(np.arange(256,dtype=np.float32)-128)/256
            abs2=(np.arange((blocks+255)//256,dtype=np.float32)%7+1)/8
            offset=np.array([.125],dtype=np.float32)
            if nested:
                absmax=((np.arange(blocks)*19+11)%256).astype(np.uint8)
                block_scale=(code2[absmax]*np.repeat(abs2,256)[:blocks]+offset[0]).astype(np.float32)
            else:
                absmax=(np.arange(blocks,dtype=np.float32)%7+1)/8
                block_scale=absmax
            oracle=code[nibbles]*np.repeat(block_scale,qblock)[:n]
            for dtype in (np.float16,np.float32):
                out=np.full(n+16,np.nan,dtype=dtype)
                for pid in range((n+2047)//2048):
                    tl.pid=pid
                    kernel(Pointer(packed),Pointer(code),Pointer(absmax),Pointer(abs2),
                           Pointer(code2),Pointer(offset),Pointer(out),n,qblock,nested,1024)
                assert np.array_equal(out[:n],oracle.astype(dtype)),(n,qblock,nested,dtype)
                assert np.isnan(out[n:]).all(), 'tail guard overwritten'
                cases.append({'n':n,'qblock':qblock,'nested':nested,'dtype':np.dtype(dtype).name})
report={'source_commit':sha,'module_sha256':hashlib.sha256(raw).hexdigest(),
        'method':'actual kernel AST evaluated with NumPy pointer model; independent unpackbits/repeat oracle',
        'cases_passed':len(cases),'all_256_packed_bytes_covered':True,
        'odd_tails_and_two_level_scale_boundaries':True,'out_of_bounds_checks':True,
        'gpu_compilation_tested':False,'gpu_floating_point_parity_tested':False,
        'training_or_quality_tested':False,'cases':cases}
target=repo/'03_validation/reports/codex_v17_cpu_layout'
target.mkdir(parents=True,exist_ok=True)
(target/'layout_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
