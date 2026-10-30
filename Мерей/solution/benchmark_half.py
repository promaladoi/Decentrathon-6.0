"""Synthetic half-input CE diagnostic, separate from the full training score."""
import argparse
import json
from pathlib import Path
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise SystemExit('Kaggle T4 required')
    from .benchmark import measure
    from .vocab import cross_entropy,_half_cross_entropy
    torch.manual_seed(173)
    report={'kind':'synthetic_half_input_ce_not_training_score','device':torch.cuda.get_device_name(0),
            'memory':'extra PyTorch allocated peak, not full nvidia-smi training peak','cases':[]}
    for rows in (256,1024,2048):
        logits=torch.randn((rows,151936),device='cuda',dtype=torch.float16,requires_grad=True)
        labels=torch.randint(151936,(rows,),device='cuda',dtype=torch.int64)
        labels[::4]=-100
        def baseline():
            return torch.autograd.grad(cross_entropy(logits.float(),labels),logits)
        def candidate():
            return torch.autograd.grad(_half_cross_entropy(logits,labels),logits)
        report['cases'].append({'rows':rows,'vocab':151936,
                                'verified_explicit_float_upcast':measure(baseline,20),
                                'half_input_float32_math':measure(candidate,20)})
        del logits,labels
    args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
