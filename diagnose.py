"""Latent-signal diagnostics on validation (never test).

Greedy outputs can hide whether a model has learned *anything*: a collapsed model
may still rank classes or sentences better than chance. This script reports, from
teacher-forced logits,
  * label AUC: one-vs-rest AUC of each class margin within its criterion type;
  * evidence AUC: AUC of sentence-ID logits (gold vs non-gold) at the first evidence step.
Usage: python diagnose.py <base_dir> [checkpoint.pt]
"""
import sys, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from model import Model, encode, TOK, LABELS
from prepare import sentences
from train import read


@torch.no_grad()
def diagnose(model, rows):
    labels=[]; ys=[]; scores=[]
    for r in rows:
        prompt=encode(r); gold=LABELS.index(r["target"]["label"])
        logits,_=model(torch.tensor([prompt+[TOK[gold]]]))
        labels.append((r["input"]["type"],gold,logits[0,-2,:6].tolist()))
        ids=sorted(sentences(r["input"]["note"])); ev=set(r["target"]["evidence"])
        if ev:
            s=logits[0,-1,[7+i for i in ids]].tolist()
            ys+= [int(i in ev) for i in ids]; scores+=s
    out={"evidence_auc":roc_auc_score(ys,scores)}
    aucs=[]
    for kind,cls in [("inclusion",[0,1,4,5]),("exclusion",[2,3,4,5])]:
        for c in cls:
            y=[int(g==c) for k,g,l in labels if k==kind]
            m=[l[c]-max(l[x] for x in cls if x!=c) for k,g,l in labels if k==kind]
            if 0<sum(y)<len(y):
                out[f"{kind}:{LABELS[c]}"]=roc_auc_score(y,m); aucs.append(out[f"{kind}:{LABELS[c]}"])
    out["mean_label_auc"]=float(np.mean(aucs))
    return out


def all_models():
    """Write results/diagnostics.json for the four models discussed in the report."""
    from model import ROOT
    specs={"SmallBase":("base",None),"SmallFirst":("base","results/first_submission/pythia14m_adaptation.pt"),
           "LargeBase":("base70",None),"LargeFinal":("base70","results/adaptation.pt")}
    out={}
    for name,(base,ck) in specs.items():
        model=Model(base=base)
        if ck: model.load_state_dict(torch.load(ROOT/ck,weights_only=True),strict=False)
        out[name]=diagnose(model.eval(),read("validation")); print(name,round(out[name]["mean_label_auc"],3),round(out[name]["evidence_auc"],3),flush=True)
    (ROOT/"results/diagnostics.json").write_text(json.dumps(out,indent=2))


if __name__=="__main__":
    if sys.argv[1:]==["--all"]:
        all_models(); sys.exit()
    model=Model(base=sys.argv[1] if len(sys.argv)>1 else "base")
    if len(sys.argv)>2:
        model.load_state_dict(torch.load(sys.argv[2],weights_only=True),strict=False)
    print(json.dumps({k:round(v,3) for k,v in diagnose(model.eval(),read("validation")).items()},indent=1))
