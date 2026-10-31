"""Whole RoPE+attention+input-backward T4 comparison; not a training score.

Original chain: RoPEPad -> native current masked/GQA SDPA -> UnpadHeads.
Candidate: RoPEPacked -> native efficient varlen -> flat output view.
Both include producer kernels and all five q/lq/k/v/lv input gradients.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
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


def _difference(actual,reference):
    finite=torch.isfinite(actual)&torch.isfinite(reference)
    a,b=actual.float(),reference.float()
    relative=(a-b).norm()/b.norm().clamp(min=1e-8)
    return {'dtype':str(actual.dtype),'reference_dtype':str(reference.dtype),
            'shape':list(actual.shape),'all_finite':bool(finite.all()),
            'relative_l2':float(relative),
            'max_abs_difference':float((a-b).abs()[finite].max()) if bool(finite.any()) else None}


def _samples(call,repeats):
    events,wall=[],[]
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize();start=time.perf_counter();begin.record()
        output=call();end.record();end.synchronize()
        events.append(begin.elapsed_time(end));wall.append(1000*(time.perf_counter()-start))
        del output
    return {'cuda_event':_summary(events),'wall':_summary(wall),
            'cuda_event_values_ms':events,'wall_values_ms':wall}


def _memory(call):
    gc.collect();torch.cuda.synchronize();start=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats();result=call();torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated();end=torch.cuda.memory_allocated()
    del result;torch.cuda.synchronize()
    return {'allocated_start_bytes':start,'allocated_peak_bytes':peak,
            'extra_allocated_peak_bytes':peak-start,'end_with_result_bytes':end,
            'after_result_release_bytes':torch.cuda.memory_allocated()}


def _data(lengths,seed):
    torch.manual_seed(seed)
    batch,heads,kv_heads,dim=len(lengths),12,2,128
    sequence=max(lengths)
    rows=torch.cat([torch.arange(n)+b*sequence for b,n in enumerate(lengths)]).cuda()
    tokens=rows.numel()
    def leaf(width,scale):
        return (torch.randn(tokens,width,device='cuda',dtype=torch.float16)*scale).requires_grad_()
    q,k,v=leaf(heads*dim,.2),leaf(kv_heads*dim,.2),leaf(kv_heads*dim,.2)
    lq,lv=leaf(heads*dim,.03),leaf(kv_heads*dim,.03)
    positions=(rows%sequence).float()
    inv=1.0/(1e6**(torch.arange(0,dim,2,device='cuda').float()/dim))
    embedding=torch.cat([torch.outer(positions,inv)]*2,dim=-1)
    cos,sin=embedding.cos(),embedding.sin()
    cumulative=torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],
                             device='cuda',dtype=torch.int32)
    positions_full=torch.arange(sequence,device='cuda')
    keys=positions_full[None,:]<torch.tensor(lengths,device='cuda')[:,None]
    causal=positions_full[:,None]>=positions_full[None,:]
    all_full=all(n==sequence for n in lengths)
    mask=None if all_full else causal[None,None]&keys[:,None,None,:]
    upstream=torch.randn(tokens,heads*dim,device='cuda',dtype=torch.float16)
    args=(q,lq,k,None,v,lv,cos,sin,rows,batch,sequence,heads,kv_heads,dim)
    return args,cumulative,mask,upstream,all_full


def _gather(padded,lengths):
    return torch.cat([padded[b,:,:n,:] for b,n in enumerate(lengths)],dim=1).unsqueeze(0)


def run_tuning(rounds=3,samples=4,warmups=3,seed=4713):
    if rounds<3 or min(samples,warmups)<1:raise ValueError('use >=3 rounds and positive samples/warmups')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Kaggle T4 required; no diagnostic was run')
    from .unpad import RoPEPad,RoPEPacked,UnpadHeads
    started=time.perf_counter()
    report={'kind':'whole_packed_rope_native_attention_diagnostic_not_training_score',
        'device':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
        'torch':torch.__version__,'rounds':rounds,'samples_per_block':samples,
        'warmups_per_variant':warmups,'seed':seed,'heads':12,'kv_heads':2,'head_dim':128,
        'input_dtypes':'q/lq/k/v/lv fp16; cos/sin fp32; rows int64; cu lengths int32',
        'timing':'whole RoPE producer, native attention, real output, and backward to q/lq/k/v/lv; CUDA events plus synchronized wall',
        'memory':'extra allocated PyTorch peak, not reserved/NVML/training peak',
        'ordering':'3 rounds, each original->packed then packed->original',
        'preparation_excluded':'only synthetic inputs/upstream, rows, cumulative boundaries and dense mask constants; no candidate gather or pack copies',
        'original_policy':'GQA native math path when no padding; explicit repeated heads and dense causal+padding mask otherwise',
        'candidate_policy':'explicit equal heads, direct native efficient varlen with both max lengths and causal custom_mask_type1',
        'training_score_claim':False,'sources':{},'cases':[]}
    for filename in ('unpad.py','benchmark_packed_integration.py','test_packed_integration.py'):
        path=Path(__file__).with_name(filename)
        if path.exists():report['sources'][filename]=hashlib.sha256(path.read_bytes()).hexdigest()
    for index,lengths in enumerate(CASES):
        args,cu,mask,upstream,all_full=_data(lengths,seed+index)
        q,lq,k,_,v,lv,cos,sin,rows,batch,sequence,heads,kv_heads,dim=args
        leaves=(q,lq,k,v,lv)
        before=[t.detach().clone() for t in leaves]
        rng=torch.cuda.get_rng_state().clone()
        scale=dim**-.5
        def forward(name):
            if name=='original':
                rep=1 if all_full else heads//kv_heads
                qp,kp,vp=RoPEPad.apply(*args,rep)
                kw={'enable_gqa':True} if all_full and heads!=kv_heads else {}
                out=torch.nn.functional.scaled_dot_product_attention(qp,kp,vp,
                    attn_mask=mask,dropout_p=0.0,is_causal=all_full and sequence>1,
                    scale=scale,**kw)
                return UnpadHeads.apply(out,rows,sequence)
            qp,kp,vp=RoPEPacked.apply(*args,heads//kv_heads)
            out=torch.ops.aten._efficient_attention_forward.default(
                qp.transpose(1,2),kp.transpose(1,2),vp.transpose(1,2),None,
                cu,cu,sequence,sequence,0.0,1,True,scale=scale)[0]
            return out.reshape(rows.numel(),heads*dim)
        def call(name):
            out=forward(name)
            return out,torch.autograd.grad(out,leaves,upstream)
        case={'lengths':list(lengths),'batch':batch,'sequence':sequence,
              'real_tokens':rows.numel(),'all_full':all_full,'rounds':[],'memory':{}}
        # Mapping has no attention arithmetic; this can legitimately be exact.
        with torch.no_grad():
            old=RoPEPad.apply(*args,1 if all_full else heads//kv_heads)
            new=RoPEPacked.apply(*args,heads//kv_heads)
            expected=[]
            for tensor in old:
                packed=_gather(tensor,lengths)
                if packed.shape[1]!=heads:packed=packed.repeat_interleave(heads//kv_heads,dim=1)
                expected.append(packed)
            case['rope_mapping_bitwise_equal']=all(torch.equal(a,b) for a,b in zip(new,expected))
            del old,new,expected
        expected,reference=call('original');actual,gradients=call('packed')
        parity={'output':_difference(actual,expected),
                'gradients':{label:_difference(a,b) for label,a,b in
                             zip(('q','lq','k','v','lv'),gradients,reference)}}
        parity['passed']=(case['rope_mapping_bitwise_equal']
            and parity['output']['all_finite'] and parity['output']['relative_l2']<2e-3
            and all(x['all_finite'] and x['relative_l2']<2e-2 for x in parity['gradients'].values()))
        case['parity']=parity
        del actual,expected,gradients,reference
        if not parity['passed']:
            case['status']='numerical_failure_no_timing';report['cases'].append(case)
            continue
        with torch.no_grad():
            qp,kp,vp=RoPEPad.apply(*args,1 if all_full else heads//kv_heads)
            choice=int(torch.ops.aten._fused_sdp_choice.default(qp,kp,vp,mask,0.0,
                all_full and sequence>1,scale=scale,enable_gqa=all_full and heads!=kv_heads))
            case['original_backend']={'choice':choice,'name':{-1:'error',0:'math',1:'flash',2:'efficient',3:'cudnn'}.get(choice,str(choice)),
                'q_shape':list(qp.shape),'k_shape':list(kp.shape),
                'mask_shape':None if mask is None else list(mask.shape)}
            del qp,kp,vp
        case['packed_backend']='direct_native_efficient_Cutlass_varlen'
        for _ in range(warmups):
            for name in ('original','packed'):
                result=call(name);del result
        aggregate={name:{'cuda_event':[],'wall':[]} for name in ('original','packed')}
        for round_index in range(rounds):
            round_result={'round':round_index+1,'blocks':[]}
            for order in (('original','packed'),('packed','original')):
                block={'order':list(order),'measurements':{}}
                for name in order:
                    measured=_samples(lambda name=name:call(name),samples)
                    block['measurements'][name]=measured
                    for metric in ('cuda_event','wall'):
                        aggregate[name][metric].extend(measured[metric+'_values_ms'])
                round_result['blocks'].append(block)
            for metric in ('cuda_event','wall'):
                round_result[metric+'_speedup']=statistics.median([
                    b['measurements']['original'][metric]['median_ms']/
                    b['measurements']['packed'][metric]['median_ms'] for b in round_result['blocks']])
            case['rounds'].append(round_result)
        case['aggregate']={name:{metric:_summary(values) for metric,values in metrics.items()}
                           for name,metrics in aggregate.items()}
        for name in ('original','packed'):case['memory'][name]=_memory(lambda name=name:call(name))
        case['inputs_unchanged']=all(torch.equal(a,b) for a,b in zip(leaves,before))
        case['rng_unchanged']=torch.equal(rng,torch.cuda.get_rng_state())
        case['status']='qualified_diagnostic_only' if case['inputs_unchanged'] and case['rng_unchanged'] else 'state_change_failure'
        report['cases'].append(case)
        del args,cu,mask,upstream,q,lq,k,v,lv,cos,sin,rows,leaves,before,rng
    report['elapsed_s']=time.perf_counter()-started
    report['status']='completed_diagnostic_only' if all(c['status']=='qualified_diagnostic_only' for c in report['cases']) else 'inspect_failed_cases'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('/kaggle/working/packed_integration_diagnostic.json'))
    args=parser.parse_args();report=run_tuning()
    args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'cases':len(report['cases']),
                      'elapsed_s':report['elapsed_s'],'out':str(args.out)},indent=2))


if __name__=='__main__':main()
