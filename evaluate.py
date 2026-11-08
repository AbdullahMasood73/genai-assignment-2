"""Run once after validation checkpoint selection, saving every test prediction."""
import json,hashlib
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from model import LABELS,ROOT,Model
from train import read,save,evaluate
from benchmark import loaded
from prepare import sentences

def prf(hit,pred,gold):
    if pred==gold==0: return [1.,1.,1.]
    return [hit/pred if pred else 0.,hit/gold if gold else 0.,2*hit/(pred+gold) if pred+gold else 0.]

def metrics(pairs,labels):
    actual=[a["target"]["label"] for a,b in pairs]
    predicted=[b["label"] for a,b in pairs]
    counts=[]
    for a,b in pairs:
        gold=set(a["target"]["evidence"])
        pred=set(b["evidence"])
        counts.append([len(gold&pred),len(pred),len(gold)])
    return dict(accuracy=accuracy_score(actual,predicted),classification=classification_report(actual,predicted,labels=labels,output_dict=True,zero_division=0),labels=labels,confusion=confusion_matrix(actual,predicted,labels=labels).tolist(),evidence_micro=prf(*map(sum,zip(*counts))),evidence_macro=np.mean([prf(*c) for c in counts],axis=0).tolist(),n=len(actual))

def final():
    ck=ROOT/"results/adaptation.pt"
    digest=hashlib.sha256(ck.read_bytes()).hexdigest()
    save("freeze",dict(checkpoint_sha256=digest,selection="maximum validation six-label macro-F1; first epoch on ties",decoding="constrained greedy"))
    for split in ["validation","test"]:
        rows=read(split)
        for name,model in [("base",Model().eval()),("adapted",loaded())]:
            pred,_=evaluate(model,rows)
            save(split+"_"+name+"_predictions",pred)
            pairs=list(zip(rows,pred))
            out={"all":metrics(pairs,LABELS)}
            for kind,labels in [("inclusion",LABELS[:2]+LABELS[4:]),("exclusion",LABELS[2:])]:
                out[kind]=metrics([(a,b) for a,b in pairs if a["input"]["type"]==kind],labels)
            save(split+"_"+name+"_metrics",out)
            if split=="test" and name=="adapted":
                errors=[]
                for a,b in pairs:
                    if a["target"]!=b:
                        errors.append(dict(id=a["id"],criterion=a["input"]["criterion"],type=a["input"]["type"],sentences=sentences(a["input"]["note"]),gold=a["target"],prediction=b))
                save("errors",errors)
if __name__=="__main__": final()
