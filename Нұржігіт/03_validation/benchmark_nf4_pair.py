"""Synthetic T4 decoder microbenchmark; these timings are NOT contest scores."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo))


def main():
    import torch
    import triton
    import bitsandbytes.functional as bnbf
    from solution.paired_nf4 import dequantize_pairs
    from solution.team_vladimir import nf4
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--pairs',type=int,default=21)
    args=parser.parse_args()
    reference=getattr(nf4.dequantize_nf4,'_paired_nf4_original',nf4.dequantize_nf4)
    torch.cuda.set_device(0)
    assert 'T4' in torch.cuda.get_device_name(0)
    torch.manual_seed(173)
    cases=[]
    # Occurrences per Qwen layer: q/o, k/v, gate/up, down. These weights describe
    # the fixed architecture, not measured full-training savings.
    shapes=[((1536,1536),2),((256,1536),2),((8960,1536),2),((1536,8960),1)]
    for shape,count in shapes:
        weight=torch.randn(shape,device='cuda',dtype=torch.float16)
        packed,state=bnbf.quantize_4bit(weight,blocksize=64,
                                      compress_statistics=True,quant_type='nf4')
        del weight
        functions={'reference':reference,'paired':dequantize_pairs}
        torch.testing.assert_close(dequantize_pairs(packed,state,torch.float16),
                                   reference(packed,state,torch.float16),rtol=0,atol=0)
        for _ in range(8):
            for fn in functions.values():
                value=fn(packed,state,torch.float16)
                del value
        torch.cuda.synchronize()
        samples={name:[] for name in functions}
        wall={name:[] for name in functions}
        for pair in range(args.pairs):
            order=('reference','paired') if pair%2==0 else ('paired','reference')
            for name in order:
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                started=time.perf_counter()
                begin.record()
                value=functions[name](packed,state,torch.float16)
                end.record()
                end.synchronize()
                samples[name].append(begin.elapsed_time(end))
                wall[name].append((time.perf_counter()-started)*1000)
                del value
        row={'shape':shape,'layers_per_block':count,'bitwise_equal':True,
             'event_median_ms':{k:statistics.median(v) for k,v in samples.items()},
             'wall_median_ms':{k:statistics.median(v) for k,v in wall.items()},
             'event_samples_ms':samples}
        row['decoder_speedup']=row['event_median_ms']['reference']/row['event_median_ms']['paired']
        cases.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='event_samples_ms'}),flush=True)
    report={'meaning':'synthetic decoder-only microbenchmark, NOT training speed or score',
            'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'triton':triton.__version__,
            'reference_file_sha256':hashlib.sha256(Path(nf4.__file__).read_bytes()).hexdigest(),
            'pairs_per_shape':args.pairs,'cases':cases}
    weighted={name:sum(c['event_median_ms'][name]*c['layers_per_block'] for c in cases)
              for name in ('reference','paired')}
    report['architecture_weighted_decoder_speedup']=weighted['reference']/weighted['paired']
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print('Decoder-only weighted ratio:',report['architecture_weighted_decoder_speedup'],flush=True)


if __name__=='__main__':
    main()
