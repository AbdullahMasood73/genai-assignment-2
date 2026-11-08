"""Sequence-level distillation of the Pythia-14M draft toward the adapted target.

Speculative decoding only needs the draft to *agree with the target*, not with
the experts. The draft is therefore trained on the target's own constrained
greedy outputs for training-split prompts (no validation/test text is used).
Usage: python distill.py [epochs]
"""
import sys, json, random, time
import torch
from torch.nn import functional as F
from model import ROOT, Model, TOK, END, LABELS, encode, choices, greedy, deployment, load_adapted
from prepare import sentences
from train import read


def targets(target, rows, end_bias):
    out=[]
    for r in rows:
        ids=sorted(sentences(r["input"]["note"]))
        seq=greedy(target,encode(r),lambda o,k=r["input"]["type"]: choices(k,ids,o),end_bias=end_bias)
        out.append(seq if seq[-1]==END else seq+[END])
    return out


@torch.no_grad()
def agreement(target, draft, rows, end_bias):
    """Fraction of positions where the draft's greedy symbol equals the target's (teacher-forced)."""
    hit=n=0
    for r,seq in zip(rows,targets(target,rows,end_bias)):
        ids=sorted(sentences(r["input"]["note"])); p=encode(r)
        logits,_=draft(torch.tensor([p+[TOK[s] for s in seq[:-1]]]))
        for j,s in enumerate(seq):
            allowed=choices(r["input"]["type"],ids,seq[:j])
            hit+=int(allowed[int(logits[0,len(p)-1+j,allowed].argmax())]==s); n+=1
    return hit/n


def main(epochs=4):
    torch.manual_seed(42); random.seed(42); torch.set_num_threads(int(sys.argv[2]) if len(sys.argv)>2 else 6)
    dep=deployment()
    target=load_adapted(dep["target"])
    train,val=read("train"),read("validation")
    seqs=targets(target,train,dep["end_bias"])
    draft=Model(base="base")
    params=[p for p in draft.parameters() if p.requires_grad]
    optim=torch.optim.AdamW(params,lr=5e-4,weight_decay=0.01)
    log=[dict(epoch=0,validation_agreement=agreement(target,draft,val[:80],dep["end_bias"]))]
    print(log[-1],flush=True)
    order=list(range(len(train)))
    for epoch in range(1,epochs+1):
        random.shuffle(order); begin=time.perf_counter(); losses=[]
        for i,k in enumerate(order):
            p=encode(train[k]); seq=seqs[k]
            logits,_=draft(torch.tensor([p+[TOK[s] for s in seq[:-1]]]))
            loss=F.cross_entropy(logits[0,len(p)-1:],torch.tensor(seq))
            (loss/4).backward(); losses.append(loss.item())
            if (i+1)%4==0 or i+1==len(order):
                torch.nn.utils.clip_grad_norm_(params,1.); optim.step(); optim.zero_grad()
        draft.eval()
        log.append(dict(epoch=epoch,loss=sum(losses)/len(losses),seconds=time.perf_counter()-begin,
                        validation_agreement=agreement(target,draft,val[:80],dep["end_bias"])))
        draft.train(); print(log[-1],flush=True)
    torch.save({k:v.detach() for k,v in draft.named_parameters() if v.requires_grad},ROOT/"results/draft14m.pt")
    (ROOT/"results/distill_log.json").write_text(json.dumps(log,indent=2))


if __name__=="__main__": main(int(sys.argv[1]) if len(sys.argv)>1 else 4)
