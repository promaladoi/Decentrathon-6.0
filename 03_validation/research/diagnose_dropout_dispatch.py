"""Diagnostic only: stop at first real rematerialized projection, no score."""
import argparse
import inspect
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import threading

p=argparse.ArgumentParser()
p.add_argument('--config',required=True)
p.add_argument('--report',required=True,type=Path)
args=p.parse_args()
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo))
import solution
native_apply=solution.apply
record={'kind':'intentionally interrupted first-forward dispatch and memory diagnosis',
        'full_task_admission':False,'completed_training':False}

def gpu_memory():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used',
        '--format=csv,noheader,nounits'],text=True)
    return sum(int(row) for row in rows.splitlines())

def apply():
    import torch
    record['mandatory_harness_context_nvml_mib']=gpu_memory()
    native_apply()
    from solution import direct_dropout,private_scope
    original=direct_dropout.try_gather
    def diagnostic(x,rows,batch,sequence,probability):
        scope=private_scope._SCOPE.get()
        threads=[]
        for thread in threading.enumerate():
            target=getattr(thread,'_target',None)
            function=getattr(target,'__func__',target)
            entry=dict(name=thread.name,class_module=type(thread).__module__,
                class_name=type(thread).__qualname__,target_module=getattr(function,'__module__',None),
                target_name=getattr(function,'__qualname__',None))
            if function is not None:
                try:entry['target_source_sha256']=hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
                except (OSError,TypeError):pass
            threads.append(entry)
        record.update(scope_present=scope is not None,
            loss_scale_available=scope is not None and scope.loss_scale is not None,
            width=x.shape[1],real_rows=x.shape[0],batch=batch,sequence=sequence,p=probability,
            torch=torch.__version__,triton=direct_dropout.triton.__version__,
            known_rng_threads=direct_dropout._known_rng_threads(),threads=threads,
            capturing=torch.cuda.is_current_stream_capturing(),first_projection_nvml_mib=gpu_memory(),
            torch_allocated_mib=torch.cuda.memory_allocated()/2**20,
            torch_reserved_mib=torch.cuda.memory_reserved()/2**20,
            generator_offset=torch.cuda.default_generators[0].get_offset())
        if scope is not None:
            buckets={};seen=set()
            for parameter in scope.base.parameters():
                if parameter.is_cuda:
                    storage=parameter.untyped_storage()
                    if storage.data_ptr() not in seen:
                        seen.add(storage.data_ptr())
                        key=str(parameter.dtype)
                        buckets[key]=buckets.get(key,0)+storage.nbytes()
            record['registered_cuda_parameter_storage_bytes_by_dtype']=buckets
            record['registered_cuda_parameter_storage_total_mib']=sum(buckets.values())/2**20
            c=scope.base.config
            h,inter,layers,vocab=c.hidden_size,c.intermediate_size,c.num_hidden_layers,c.vocab_size
            dim=getattr(c,'head_dim',None) or h//c.num_attention_heads
            matrices=layers*(h*dim*(2*c.num_attention_heads+2*c.num_key_value_heads)+3*h*inter)
            embeddings=vocab*h
            record['config_dimensions']=dict(hidden=h,intermediate=inter,layers=layers,vocabulary=vocab,
                heads=c.num_attention_heads,kv_heads=c.num_key_value_heads,head_dim=dim)
            record['matrix_nf4_plus_fp16_embedding_payload_lower_bound_mib']=(matrices/2+embeddings*2)/2**20
            record['all_matrix_and_embedding_4bit_payload_lower_bound_mib']=(matrices+embeddings)/2/2**20
        record['fast_gather_selected']=original(x,rows,batch,sequence,probability) is not None
        args.report.write_text(json.dumps(record,indent=2)+'\n')
        print('FIRST_FORWARD_DIAGNOSIS',json.dumps(record),flush=True)
        raise RuntimeError('Intentional diagnostic stop; no completed-task result')
    direct_dropout.try_gather=diagnostic

solution.apply=apply
sys.argv=['harness/_train.py','--config',args.config,'--metrics',str(args.report.with_suffix('.unused.json')),'--solution']
try:runpy.run_path(str(repo/'harness/_train.py'),run_name='__main__')
except BaseException:
    if not args.report.is_file():raise
print('DIAGNOSTIC_STOP_CONFIRMED_NO_ADMISSION',flush=True)
