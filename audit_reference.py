"""Optional audit: compare the from-scratch forward pass with Hugging Face Transformers.

Requires transformers==4.56.2 (not needed by the application), e.g.
    pip install --target audit_env transformers==4.56.2
    set PYTHONPATH=audit_env && python audit_reference.py
Compares restricted-vocabulary logits of the *unadapted* checkpoints at three lengths.
"""
import json
import torch
from transformers import GPTNeoXForCausalLM
from model import Model, ROOT, TOK, encode
from train import read


@torch.no_grad()
def audit():
    torch.set_num_threads(1)
    summary=[]
    row=read("validation")[0]
    for base,name in [("base","Pythia-14M"),("base70","Pythia-70M")]:
        reference=GPTNeoXForCausalLM.from_pretrained(str(ROOT/base),dtype=torch.float32,attn_implementation="eager").eval()
        custom=Model(base=base).eval()
        for n in [5,32,96]:
            ids=torch.tensor([encode(row)[:n]])
            official=reference(ids).logits[...,TOK]
            actual=custom(ids)[0]
            summary.append({"model":name,"length":n,"max_abs_logit_error":(actual-official).abs().max().item(),
                            "mean_abs_logit_error":(actual-official).abs().mean().item(),
                            "same_argmax_positions":(actual.argmax(-1)==official.argmax(-1)).sum().item()})
            print(summary[-1],flush=True)
    (ROOT/"results/reference_audit.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__": audit()
