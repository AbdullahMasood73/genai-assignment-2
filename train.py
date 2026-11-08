"""Supervised adaptation on the training split; checkpoint selection on validation only.

Objective: teacher-forced cross-entropy over the answer symbols only
(label code, ascending evidence sentence IDs, stop). Two optional re-weightings
address the observed collapse to majority labels and empty evidence:
  * --class-weight balanced : label-token loss weighted by (N / (K * n_c))**power,
    computed from training-split label frequencies only.
  * --evidence-weight w     : multiplies the loss on evidence-ID positions (not stop),
    counteracting the 58% of training records whose answer stops right after the label.
"""
import argparse, json, random, time, collections
from pathlib import Path
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score
from prepare import sentences
from model import ROOT, Model, LABELS, TOK, END, encode, choices, greedy


def read(split):
    return [json.loads(line) for line in (ROOT/"data"/(split+".jsonl")).read_text(encoding="utf-8").splitlines()]


def predict(model, item, end_bias=0.0):
    i=item["input"]
    ids=sorted(sentences(i["note"]))
    allowed=lambda out: choices(i["type"],ids,out)
    seq=greedy(model,encode(item),allowed,end_bias=end_bias)
    return {"label":LABELS[seq[0]],"evidence":[s-7 for s in seq[1:] if s!=END]}


def evidence_f1(data, preds):
    hit=sum(len(set(r["target"]["evidence"])&set(p["evidence"])) for r,p in zip(data,preds))
    n_pred=sum(len(p["evidence"]) for p in preds)
    n_gold=sum(len(set(r["target"]["evidence"])) for r in data)
    return 2*hit/(n_pred+n_gold) if n_pred+n_gold else 1.0


def evaluate(model, data, end_bias=0.0):
    model.eval()
    preds=[predict(model,row,end_bias) for row in data]
    gold=[r["target"]["label"] for r in data]
    labels=[r["label"] for r in preds]
    present=sorted(set(gold))
    stats={"accuracy":accuracy_score(gold,labels),
           "macro_f1":f1_score(gold,labels,labels=LABELS,average="macro",zero_division=0),
           "macro_f1_present":f1_score(gold,labels,labels=present,average="macro",zero_division=0),
           "evidence_micro_f1":evidence_f1(data,preds)}
    # Validation-only selection score: equal weight on labels and evidence.
    stats["selection"]=(stats["macro_f1_present"]+stats["evidence_micro_f1"])/2
    return preds,stats


def save(name,x,folder="results"):
    (ROOT/folder).mkdir(exist_ok=True)
    (ROOT/folder/(name+".json")).write_text(json.dumps(x,indent=2))


def label_weights(trainset, mode, power=0.5):
    if mode=="none":
        return torch.ones(len(TOK))
    counts=collections.Counter(r["target"]["label"] for r in trainset)
    w=torch.ones(len(TOK))
    for i,l in enumerate(LABELS):
        w[i]=(len(trainset)/(len(LABELS)*counts[l]))**power
    return w


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    out=args.out
    trainset,val=read("train"),read("validation")
    model=Model(base=args.base)
    config={k:v for k,v in vars(args).items()}
    config["trainable_parameters"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["total_parameters"]=sum(p.numel() for p in model.parameters())
    save("train_config",config,out)
    weights=label_weights(trainset,args.class_weight,args.balance_power)
    config["label_weights"]=weights[:6].tolist()
    save("train_config",config,out)
    pred,stats=evaluate(model,val)
    save("validation_base",pred,out)
    history=[dict(epoch=0,**stats)]
    print(history[-1],flush=True)
    params=[p for p in model.parameters() if p.requires_grad]
    optim=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)
    steps=args.epochs*((len(trainset)+args.accumulate-1)//args.accumulate)
    sched=torch.optim.lr_scheduler.LambdaLR(optim,lambda s: min(1.0,(s+1)/max(1,int(0.06*steps)))*max(0.0,(steps-s)/steps)) if args.schedule=="linear" else None
    best=-1
    for epoch in range(1,args.epochs+1):
        model.train()
        random.shuffle(trainset)
        losses=[]
        begin=time.perf_counter()
        for i,row in enumerate(trainset):
            prompt=encode(row)
            gold=[LABELS.index(row["target"]["label"])]+[7+j for j in sorted(set(row["target"]["evidence"]))]+[END]
            logits,_=model(torch.tensor([prompt+[TOK[s] for s in gold[:-1]]]))
            per=F.cross_entropy(logits[0,len(prompt)-1:],torch.tensor(gold),reduction="none")
            w=torch.ones(len(gold))
            w[0]=weights[gold[0]]
            w[1:-1]=args.evidence_weight      # evidence IDs only; stop keeps weight 1
            loss=(per*w).sum()/w.sum()
            (loss/args.accumulate).backward()
            if (i+1)%args.accumulate==0 or i+1==len(trainset):
                torch.nn.utils.clip_grad_norm_(params,1.)
                optim.step()
                if sched: sched.step()
                optim.zero_grad()
            losses.append(per.mean().item())
        train_time=time.perf_counter()-begin
        pred,stats=evaluate(model,val)
        history.append(dict(epoch=epoch,loss=sum(losses)/len(losses),train_seconds=train_time,**stats))
        save("history",history,out)
        print(history[-1],flush=True)
        torch.save({k:v.detach() for k,v in model.named_parameters() if v.requires_grad},ROOT/out/"last.pt")
        if stats["selection"]>best:
            best=stats["selection"]
            torch.save({k:v.detach() for k,v in model.named_parameters() if v.requires_grad},ROOT/out/"adaptation.pt")
            save("validation_best",pred,out)
            save("selected",history[-1],out)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",default="base")
    ap.add_argument("--epochs",type=int,default=4)
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--accumulate",type=int,default=4)
    ap.add_argument("--class-weight",choices=["none","balanced"],default="none")
    ap.add_argument("--balance-power",type=float,default=0.5)
    ap.add_argument("--evidence-weight",type=float,default=1.0)
    ap.add_argument("--schedule",choices=["constant","linear"],default="constant")
    ap.add_argument("--threads",type=int,default=2)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default="results")
    train(ap.parse_args())
