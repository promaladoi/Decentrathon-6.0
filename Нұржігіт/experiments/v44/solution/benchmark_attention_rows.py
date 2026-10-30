"""Balanced T4 native-attention diagnostics; not an official training score."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback
import warnings
import torch


CASES=((512,128,96,64),(512,512,512,512),(256,192,128,96),(128,64,32,16))


def _summary(values):
    ordered=sorted(values)
    def percentile(p):
        point=(len(ordered)-1)*p;lo=int(point);hi=min(lo+1,len(ordered)-1)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(point-lo)
    return {'samples':len(values),'median_ms':statistics.median(values),
            'q25_ms':percentile(.25),'q75_ms':percentile(.75),
            'minimum_ms':min(values),'maximum_ms':max(values)}


def _samples(call,repeats):
    cuda,wall=[],[]
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize();before=time.perf_counter();begin.record()
        result=call();end.record();end.synchronize()
        cuda.append(begin.elapsed_time(end));wall.append(1000*(time.perf_counter()-before))
        del result
    return {'cuda_event':_summary(cuda),'wall':_summary(wall),
            'cuda_event_values_ms':cuda,'wall_values_ms':wall}


def _memory(call):
    gc.collect();torch.cuda.synchronize();start=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats();result=call();torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated();end=torch.cuda.memory_allocated()
    del result;torch.cuda.synchronize()
    return {'allocated_start_bytes':start,'allocated_peak_bytes':peak,
            'extra_allocated_peak_bytes':peak-start,'end_with_result_bytes':end,
            'after_result_release_bytes':torch.cuda.memory_allocated()}


def _difference(actual,expected):
    finite=torch.isfinite(actual)&torch.isfinite(expected)
    rel=(actual.float()-expected.float()).norm()/expected.float().norm().clamp(min=1e-8)
    return {'shape':list(actual.shape),'dtype':str(actual.dtype),
            'reference_dtype':str(expected.dtype),
            'bitwise_equal':bool(torch.equal(actual.view(torch.int16),expected.view(torch.int16)))
                if actual.dtype==expected.dtype==torch.float16 else bool(torch.equal(actual,expected)),
            'relative_l2':float(rel),'all_finite':bool(finite.all()),
            'max_abs_difference':float((actual.float()-expected.float()).abs()[finite].max())
                if bool(finite.any()) else None}


def _canonical_grad(gradient,kv_heads):
    if gradient.shape[1]==kv_heads:return gradient.float()
    b,h,s,d=gradient.shape
    return gradient.float().reshape(b,kv_heads,h//kv_heads,s,d).sum(dim=2)


def _backend(q,k,v,mask,is_causal,scale,enable_gqa=False):
    params=torch.backends.cuda.SDPAParams(q,k,v,mask,0.0,is_causal,enable_gqa)
    names={-1:'error',0:'math',1:'flash',2:'efficient',3:'cudnn'}
    choice=int(torch.ops.aten._fused_sdp_choice.default(q,k,v,mask,0.0,is_causal,
                                                      scale=scale,enable_gqa=enable_gqa))
    return {'choice':choice,'choice_name':names.get(choice,str(choice)),
            'efficient_eligible':bool(torch.backends.cuda.can_use_efficient_attention(params,debug=False)),
            'flash_eligible':bool(torch.backends.cuda.can_use_flash_attention(params,debug=False)),
            'query_shape':list(q.shape),'key_shape':list(k.shape),'query_stride':list(q.stride()),
            'key_stride':list(k.stride()),'mask_shape':None if mask is None else list(mask.shape),
            'is_causal':is_causal,'enable_gqa':enable_gqa}


def run_tuning(rounds=3,samples=4,warmups=3,seed=4407):
    if rounds<3 or samples<1 or warmups<1:raise ValueError('use >=3 rounds and positive samples/warmups')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Kaggle T4 required; no benchmark was run')
    from .attention_rows import make_layout,VARIANTS
    started=time.perf_counter()
    report={'kind':'synthetic_native_attention_not_training_score',
        'device':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
        'torch':torch.__version__,'seed':seed,'rounds':rounds,'samples_per_block':samples,
        'warmups_per_variant':warmups,'batch':4,'query_heads':12,'key_heads':2,'head_dim':128,
        'timing':'native attention plus real-token output and all padded-leaf Q/K/V gradients; CUDA events and synchronized wall',
        'memory':'extra PyTorch allocated peak per call, not NVML/reserved/training peak',
        'ordering':'each candidate vs current in AB then BA order in each of three rounds',
        'preparation_excluded':'length metadata, masks, cumulative int32 boundaries and explicit K/V repeats prepared once outside timing, as once-per-batch integration would do',
        'gather_limit':'all variants use native cat/slice real-token output; padded baseline training uses its own one-launch gather instead; packed includes three input gathers not needed by future fused-pack integration',
        'training_score_claim':False,'sources':{},'cases':[]}
    for filename in ('attention_rows.py','test_attention_rows.py','benchmark_attention_rows.py','unpad.py'):
        path=Path(__file__).with_name(filename)
        if path.exists():report['sources'][filename]=hashlib.sha256(path.read_bytes()).hexdigest()
    torch.manual_seed(seed)
    for lengths in CASES:
        sequence=max(lengths);layout=make_layout(lengths,sequence,'cuda');scale=128**-0.5
        q=(torch.randn(4,12,sequence,128,device='cuda',dtype=torch.float16)*.2).requires_grad_()
        ksmall=(torch.randn(4,2,sequence,128,device='cuda',dtype=torch.float16)*.2).requires_grad_()
        vsmall=(torch.randn_like(ksmall)*.2).requires_grad_()
        k=ksmall.detach().repeat_interleave(6,dim=1).requires_grad_()
        v=vsmall.detach().repeat_interleave(6,dim=1).requires_grad_()
        upstream=torch.randn(layout.total,12*128,device='cuda',dtype=torch.float16)
        originals=[t.detach().clone() for t in (q,ksmall,vsmall,k,v)]
        def inputs(name):
            return (q,ksmall,vsmall) if name=='current' and layout.all_full else (q,k,v)
        def call(name,force=False):
            tensors=inputs(name)
            output=VARIANTS[name](*tensors,layout,scale,force_efficient=force)
            gradients=torch.autograd.grad(output,tensors,upstream)
            return output,gradients
        case={'lengths':list(lengths),'padded_sequence':sequence,'real_tokens':layout.total,
              'pair_work_dense':4*sequence*sequence,
              'pair_work_real_causal':sum(n*(n+1)//2 for n in lengths),
              'all_full':layout.all_full,'parity':{},'backends':{},'forced_efficient':{},
              'rounds':[],'memory':{}}
        iq,ik,iv=inputs('current')
        case['backends']['current']=_backend(iq,ik,iv,layout.dense_mask,
            layout.all_full and sequence>1,scale,ik.shape[1]!=iq.shape[1])
        case['backends']['equalheads']=_backend(q,k,v,layout.dense_mask,layout.all_full and sequence>1,scale)
        case['backends']['rows']=[_backend(q[b:b+1,:,:n],k[b:b+1,:,:n],v[b:b+1,:,:n],None,n>1,scale)
                                 for b,n in enumerate(lengths)]
        case['backends']['packed']={'choice_name':'direct_native_efficient_Cutlass_varlen',
            'query_shape':[1,layout.total,12,128],'explicit_max_q':layout.maximum,
            'explicit_max_k':layout.maximum,'cumulative_dtype':'int32','custom_mask_type':1}
        case['backends']['causal_bias']=_backend(q,k,v,None if layout.all_full else layout.padding_bias,
                                                 sequence>1,scale)
        expected,expected_grad=call('current')
        reference=(expected_grad[0].float(),_canonical_grad(expected_grad[1],2),
                   _canonical_grad(expected_grad[2],2))
        qualified=[]
        for name in ('equalheads','rows','packed','causal_bias'):
            try:
                actual,gradients=call(name)
                canonical=(gradients[0].float(),_canonical_grad(gradients[1],2),
                           _canonical_grad(gradients[2],2))
                parity={'output':_difference(actual,expected),
                    'gradients':{label:_difference(a,b) for label,a,b in zip(('q','k_grouped','v_grouped'),canonical,reference)}}
                parity['passed']=(parity['output']['all_finite'] and parity['output']['relative_l2']<3e-3
                    and all(x['all_finite'] and x['relative_l2']<1e-2 for x in parity['gradients'].values()))
                case['parity'][name]=parity
                if parity['passed']:qualified.append(name)
                del actual,gradients,canonical
            except Exception as exc:
                case['parity'][name]={'passed':False,'error':f'{type(exc).__name__}: {exc}',
                                      'traceback':traceback.format_exc()}
        del expected,expected_grad,reference
        for name in ('current',*qualified):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');forced=call(name,True)
                del forced
                case['forced_efficient'][name]={'completed':True,'warnings':[str(x.message) for x in caught]}
            except Exception as exc:
                case['forced_efficient'][name]={'completed':False,'error':f'{type(exc).__name__}: {exc}'}
        for _ in range(warmups):
            for name in ('current',*qualified):
                result=call(name);del result
        torch.cuda.synchronize()
        aggregate={name:{'cuda_event':[],'wall':[]} for name in ('current',*qualified)}
        for index in range(rounds):
            round_result={'round':index+1,'pairs':[]}
            for candidate in qualified:
                pair={'candidate':candidate,'blocks':[]}
                for order in (('current',candidate),(candidate,'current')):
                    block={'order':list(order),'measurements':{}}
                    for name in order:
                        measured=_samples(lambda name=name:call(name),samples)
                        block['measurements'][name]=measured
                        for metric in ('cuda_event','wall'):
                            aggregate[name][metric].extend(measured[metric+'_values_ms'])
                    pair['blocks'].append(block)
                for metric in ('cuda_event','wall'):
                    pair[metric+'_speedup']=statistics.median([
                        b['measurements']['current'][metric]['median_ms']/
                        b['measurements'][candidate][metric]['median_ms'] for b in pair['blocks']])
                round_result['pairs'].append(pair)
            case['rounds'].append(round_result)
        case['aggregate']={name:{metric:_summary(values) for metric,values in data.items()}
                           for name,data in aggregate.items()}
        for name in ('current',*qualified):case['memory'][name]=_memory(lambda name=name:call(name))
        case['inputs_unchanged']=all(torch.equal(a,b) for a,b in zip((q,ksmall,vsmall,k,v),originals))
        case['status']='qualified_diagnostic_only' if len(qualified)==4 and case['inputs_unchanged'] else 'partial_or_numerical_failure'
        report['cases'].append(case)
        del q,ksmall,vsmall,k,v,upstream,originals,layout
    report['elapsed_s']=time.perf_counter()-started
    report['status']='completed_diagnostic_only' if all(c['status']=='qualified_diagnostic_only' for c in report['cases']) else 'inspect_failed_variants'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('/kaggle/working/attention_rows_diagnostic.json'))
    args=parser.parse_args()
    report=run_tuning();args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'cases':len(report['cases']),
                      'elapsed_s':report['elapsed_s'],'out':str(args.out)},indent=2))


if __name__=='__main__':main()
