"""CPU address/bit-layout check of actual kernel bodies, not Triton GPU codegen.

The independent oracle uses NumPy packbits/unpackbits. Dyadic values isolate
layout and tail safety. Actual GPU output/gradient/RNG tests remain mandatory.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

repo = Path(__file__).resolve().parents[1]
sha = subprocess.check_output(['git', 'rev-parse', sys.argv[1] if len(sys.argv) > 1 else 'HEAD'],
                              cwd=repo, text=True).strip()
raw = subprocess.check_output(['git', 'show', sha+':solution/compact_dropout.py'], cwd=repo)


class Vector(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Vector)


def vector(x):
    return np.asarray(x).view(Vector)


class Pointer:
    def __init__(self, values, offset=0):
        self.values, self.offset = values, offset
        self.dtype = SimpleNamespace(element_ty=values.dtype)

    def __add__(self, offset):
        return Pointer(self.values, self.offset+offset)


class Language:
    float32, int32, uint8 = np.float32, np.int32, np.uint8
    pid = 0

    def program_id(self, axis):
        assert axis == 0
        return vector(np.int32(self.pid))

    def arange(self, begin, end):
        return vector(np.arange(begin, end, dtype=np.int32))

    def load(self, pointer, mask=True, other=0):
        i, mask = np.broadcast_arrays(np.asarray(pointer.offset), np.asarray(mask))
        assert not np.any(mask & ((i < 0) | (i >= pointer.values.size))), 'out-of-bounds read'
        safe = np.clip(i, 0, pointer.values.size-1).astype(np.int64)
        return vector(np.where(mask, pointer.values[safe], np.array(other, dtype=pointer.values.dtype)))

    def store(self, pointer, values, mask=True):
        i, values, mask = np.broadcast_arrays(np.asarray(pointer.offset), np.asarray(values), np.asarray(mask))
        assert not np.any(mask & ((i < 0) | (i >= pointer.values.size))), 'out-of-bounds write'
        pointer.values[i[mask].astype(np.int64)] = values[mask]

    def reshape(self, x, shape):
        return vector(x.reshape(shape))

    def sum(self, x, axis):
        return vector(np.sum(x, axis=axis, dtype=np.int32))


tl = Language()
scope = {'tl': tl}
names = ('compact_dropout_pack_rows', 'compact_dropout_unpack_gradient')
nodes = [copy.deepcopy(n) for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(nodes) == 2
for node in nodes:
    node.decorator_list, node.returns = [], None
    for arg in node.args.args:
        arg.annotation = None
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
             '<actual packed dropout kernel bodies>', 'exec'), scope)
rng = np.random.default_rng(211)
cases = []


def check(full, rows, n, dtype, name):
    hidden = full.shape[1]
    selected = full[rows].reshape(-1)[:n]
    oracle_bytes = np.packbits(selected, bitorder='little')
    x = ((np.arange(n) % 31 - 15)/16).astype(dtype)
    grad = ((np.arange(n) % 19 - 9)/8).astype(dtype)
    out = np.full(n+16, -32, dtype=dtype)
    dx = np.full(n+16, -64, dtype=dtype)
    packed = np.full(oracle_bytes.size+16, 204, dtype=np.uint8)
    for pid in range((n+1023)//1024):
        tl.pid = pid
        scope[names[0]](Pointer(x), Pointer(full.reshape(-1)), Pointer(rows.copy()),
                        Pointer(out), Pointer(packed), n, hidden, 1.25, 1024)
    assert np.array_equal(packed[:oracle_bytes.size], oracle_bytes)
    assert (packed[oracle_bytes.size:] == 204).all(), 'packed tail overwritten'
    expected = (x.astype(np.float32)*selected*np.float32(1.25)).astype(dtype)
    assert np.array_equal(out[:n], expected)
    assert (out[n:] == -32).all(), 'output tail overwritten'
    for pid in range((n+1023)//1024):
        tl.pid = pid
        scope[names[1]](Pointer(grad), Pointer(packed[:oracle_bytes.size]), Pointer(dx), n, 1.25, 1024)
    unpacked = np.unpackbits(oracle_bytes, bitorder='little')[:n]
    expected_dx = (grad.astype(np.float32)*unpacked*np.float32(1.25)).astype(dtype)
    assert np.array_equal(dx[:n], expected_dx)
    assert (dx[n:] == -64).all(), 'gradient tail overwritten'
    cases.append({'name': name, 'n': n, 'hidden': hidden, 'dtype': np.dtype(dtype).name})


for hidden in (7, 17, 65, 128, 1536):
    for count in (1, 5, 17, 32):
        full = rng.integers(0, 2, (count*3, hidden), dtype=np.uint8).astype(bool)
        rows = np.arange(count*3, dtype=np.int64)[::3]
        for removed in (0, 1, 3):
            n = count*hidden-removed
            for dtype in (np.float16, np.float32):
                check(full, rows, n, dtype, 'strided real rows and partial final byte')
all_bytes = np.unpackbits(np.arange(256, dtype=np.uint8), bitorder='little').reshape(256, 8).astype(bool)
for n in (2048, 2045):
    for dtype in (np.float16, np.float32):
        check(all_bytes, np.arange(256, dtype=np.int64), n, dtype, 'all 256 byte patterns')
report = {'source_commit': sha, 'module_sha256': hashlib.sha256(raw).hexdigest(),
          'method': 'actual kernel AST with bounds-checked NumPy pointer model; independent packbits/unpackbits oracle',
          'cases_passed': len(cases), 'all_256_byte_patterns': True,
          'gpu_compilation_tested': False, 'gpu_rounding_or_rng_tested': False,
          'training_quality_or_speed_tested': False, 'cases': cases}
target = repo/'03_validation/reports/codex_v20_cpu_layout'
target.mkdir(parents=True, exist_ok=True)
(target/'layout_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))
