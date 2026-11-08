"""Optional reference audit. Requires transformers==4.56.2 in a separate env."""
import json
import torch
from transformers import GPTNeoXForCausalLM
from model import Model,ROOT,TOK
from train import read
from model import encode
@torch.no_grad()
def audit():
    reference=GPTNeoXForCausalLM.from_pretrained(str(ROOT/"base"),dtype=torch.float32,attn_implementation="eager").eval()
    custom=Model().eval()
    summary=[]
    for n in [5,32,96]:
        ids=torch.tensor([encode(read("validation")[0])[:n]])
        official=reference(ids).logits[...,TOK]
        actual=custom(ids)[0]
        summary.append({"length":n,"max_abs_logit_error":(actual-official).abs().max().item(),"mean_abs_logit_error":(actual-official).abs().mean().item(),"same_argmax_positions":(actual.argmax(-1)==official.argmax(-1)).sum().item()})
    print(summary)
    (ROOT/"results/reference_audit.json").write_text(json.dumps(summary,indent=2))
if __name__=="__main__":audit()
