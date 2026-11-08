import json, time, random
from pathlib import Path
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score
from prepare import sentences
from model import ROOT, Model, LABELS, TOK, END, encode, choices, greedy

def read(split):
    return [json.loads(line) for line in (ROOT/"data"/(split+".jsonl")).read_text().splitlines()]

def predict(model,item):
    i=item["input"]
    allowed=lambda out: choices(i["type"],list(sentences(i["note"])),out)
    seq=greedy(model,encode(item),allowed)
    return {"label":LABELS[seq[0]],"evidence":[s-7 for s in seq[1:] if s!=END]}

def evaluate(model, data):
    model.eval()
    preds=[predict(model,row) for row in data]
    gold=[r["target"]["label"] for r in data]
    labels=[r["label"] for r in preds]
    return preds,{"accuracy":accuracy_score(gold,labels),"macro_f1":f1_score(gold,labels,labels=LABELS,average="macro",zero_division=0)}

def save(name,x):
    (ROOT/"results"/(name+".json")).write_text(json.dumps(x,indent=2))

def train():
    torch.manual_seed(42)
    random.seed(42)
    (ROOT/"results").mkdir(exist_ok=True)
    trainset,val=read("train"),read("validation")
    model=Model()
    pred,stats=evaluate(model,val)
    save("validation_base",pred)
    history=[dict(epoch=0,**stats)]
    params=[p for p in model.parameters() if p.requires_grad]
    optim=torch.optim.AdamW(params,lr=0.0002)
    best=-1
    for epoch in range(1,5):
        model.train()
        random.shuffle(trainset)
        losses=[]
        for i,row in enumerate(trainset):
            prompt=encode(row)
            gold=[LABELS.index(row["target"]["label"])]+[7+j for j in sorted(set(row["target"]["evidence"]))]+[END]
            logits,_=model(torch.tensor([prompt+[TOK[s] for s in gold[:-1]]]))
            loss=F.cross_entropy(logits[0,len(prompt)-1:],torch.tensor(gold))
            (loss/4).backward()
            if (i+1)%4==0 or i+1==len(trainset):
                torch.nn.utils.clip_grad_norm_(params,1.)
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        pred,stats=evaluate(model,val)
        history.append(dict(epoch=epoch,loss=sum(losses)/len(losses),**stats))
        save("history",history)
        print(history[-1],flush=True)
        if stats["macro_f1"]>best:
            best=stats["macro_f1"]
            torch.save({k:v.detach() for k,v in model.named_parameters() if v.requires_grad},ROOT/"results/adaptation.pt")
            save("validation_best",pred)
            save("selected",history[-1])

if __name__=="__main__": train()
