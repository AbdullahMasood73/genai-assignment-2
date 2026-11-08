"""Calibrate on validation, freeze, then run the single final test evaluation.

    python evaluate.py calibrate   # validation only: choose stop-symbol bias, write deployment.json
    python evaluate.py final       # write freeze.json, then score validation and test

Systems compared on identical rows with identical constrained-greedy grammar:
  majority  : type-conditioned training-split majority label, no evidence (trivial baseline)
  lead      : majority label + first note sentence as evidence (positional baseline)
  base      : pretrained target before fine-tuning (stop bias 0)
  adapted0  : fine-tuned target, stop bias 0
  adapted   : fine-tuned target with the validation-calibrated stop bias (deployed)
"""
import json, hashlib, sys, collections, datetime
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from model import LABELS, ROOT, Model, deployment, load_adapted
from train import read, save, evaluate, predict
from prepare import sentences

TYPE_LABELS={"inclusion":LABELS[:2]+LABELS[4:],"exclusion":LABELS[2:]}


def prf(hit,pred,gold):
    """Precision/recall/F1 for one evidence set. Both empty -> 1,1,1; one side empty -> 0."""
    if pred==gold==0: return [1.,1.,1.]
    return [hit/pred if pred else 0.,hit/gold if gold else 0.,2*hit/(pred+gold) if pred+gold else 0.]


def metrics(pairs,labels):
    actual=[a["target"]["label"] for a,b in pairs]
    predicted=[b["label"] for a,b in pairs]
    present=[l for l in labels if l in actual]
    counts=[]
    for a,b in pairs:
        gold=set(a["target"]["evidence"]); pred=set(b["evidence"])
        counts.append([len(gold&pred),len(pred),len(gold)])
    nonempty=[c for c in counts if c[2]>0]
    return dict(n=len(actual),accuracy=accuracy_score(actual,predicted),
                macro_f1=f1_score(actual,predicted,labels=labels,average="macro",zero_division=0),
                macro_f1_present=f1_score(actual,predicted,labels=present,average="macro",zero_division=0),
                classification=classification_report(actual,predicted,labels=labels,output_dict=True,zero_division=0),
                labels=labels,confusion=confusion_matrix(actual,predicted,labels=labels).tolist(),
                predicted_distribution=dict(collections.Counter(predicted)),
                evidence_micro=prf(*map(sum,zip(*counts))),
                evidence_macro=np.mean([prf(*c) for c in counts],axis=0).tolist(),
                evidence_macro_nonempty_gold=np.mean([prf(*c) for c in nonempty],axis=0).tolist() if nonempty else None,
                records_with_gold_evidence=len(nonempty),
                records_with_predicted_evidence=sum(c[1]>0 for c in counts),
                empty_both=sum(c[1]==0 and c[2]==0 for c in counts))


def report(rows,pred):
    pairs=list(zip(rows,pred))
    out={"all":metrics(pairs,LABELS)}
    for kind,labels in TYPE_LABELS.items():
        out[kind]=metrics([(a,b) for a,b in pairs if a["input"]["type"]==kind],labels)
    return out


def majority(rows):
    train=read("train")
    best={k:collections.Counter(r["target"]["label"] for r in train if r["input"]["type"]==k).most_common(1)[0][0] for k in TYPE_LABELS}
    return [{"label":best[r["input"]["type"]],"evidence":[]} for r in rows]


def lead(rows):
    """Majority label plus the first sentence (usually age/sex) as evidence: a positional baseline."""
    return [dict(p,evidence=[min(sentences(r["input"]["note"]))]) for p,r in zip(majority(rows),rows)]


def summary_row(split,name,out):
    a=out["all"]
    return dict(split=split,model=name,accuracy=a["accuracy"],macro_f1=a["macro_f1"],macro_f1_present=a["macro_f1_present"],
                inclusion_accuracy=out["inclusion"]["accuracy"],exclusion_accuracy=out["exclusion"]["accuracy"],
                evidence_micro_p=a["evidence_micro"][0],evidence_micro_r=a["evidence_micro"][1],evidence_micro_f1=a["evidence_micro"][2],
                evidence_macro_f1=a["evidence_macro"][2],
                evidence_macro_f1_nonempty=a["evidence_macro_nonempty_gold"][2] if a["evidence_macro_nonempty_gold"] else None)


def baselines():
    """Add the rule-based lead-sentence baseline to an existing summary (no model is run)."""
    summary=[s for s in json.loads((ROOT/"results/metrics_summary.json").read_text()) if s["model"]!="lead"]
    for split in ["validation","test"]:
        rows=read(split); pred=lead(rows)
        save(f"{split}_lead_predictions",pred); out=report(rows,pred); save(f"{split}_lead_metrics",out)
        at=max(i for i,s in enumerate(summary) if s["split"]==split and s["model"]=="majority")
        summary.insert(at+1,summary_row(split,"lead",out))
    save("metrics_summary",summary)
    save("test_metrics_summary",[s for s in summary if s["split"]=="test"])


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def calibrate():
    """Choose the stop-symbol bias on validation only (labels are unaffected by it)."""
    dep=json.loads((ROOT/"results/deployment.json").read_text())
    model=load_adapted(dep["target"])
    rows=read("validation")
    sweep=[]
    for bias in [1,0.5,0,-0.5,-1,-1.5,-2,-2.5,-3,-3.5,-4,-5,-6]:
        pred,_=evaluate(model,rows,bias)
        m=metrics(list(zip(rows,pred)),LABELS)
        sweep.append(dict(end_bias=bias,evidence_micro=m["evidence_micro"],evidence_macro=m["evidence_macro"],
                          predicted_evidence_records=m["records_with_predicted_evidence"]))
        print(sweep[-1],flush=True)
    best=max(sweep,key=lambda s:(round(s["evidence_micro"][2],6),s["end_bias"]))
    dep["end_bias"]=best["end_bias"]
    save("calibration",dict(criterion="maximum validation evidence micro-F1; smaller |bias| on ties",sweep=sweep,selected=best["end_bias"]))
    (ROOT/"results/deployment.json").write_text(json.dumps(dep,indent=2))
    print("selected end_bias",best["end_bias"])


def final():
    dep=deployment()
    target=dep["target"]
    save("freeze",dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                       target=dep["target_name"],checkpoint=target["checkpoint"],checkpoint_sha256=sha(target["checkpoint"]),
                       base_revision=(ROOT/target["base"]/"revision.txt").read_text().strip(),
                       selection="maximum validation selection score = mean(macro-F1 over present classes, evidence micro-F1)",
                       end_bias=dep["end_bias"],decoding="constrained greedy, max 102 symbols, no sampling",
                       note="Written before the test split is scored. Nothing below is tuned on test."))
    summary=[]
    base=Model(base=target["base"]).eval()
    adapted=load_adapted(target)
    for split in ["validation","test"]:
        rows=read(split)
        systems={"majority":majority(rows),"lead":lead(rows),
                 "base":evaluate(base,rows,0.0)[0],
                 "adapted0":evaluate(adapted,rows,0.0)[0],
                 "adapted":evaluate(adapted,rows,dep["end_bias"])[0]}
        for name,pred in systems.items():
            save(f"{split}_{name}_predictions",pred)
            out=report(rows,pred)
            save(f"{split}_{name}_metrics",out)
            summary.append(summary_row(split,name,out))
            print(summary[-1],flush=True)
        if split=="test":
            pred=systems["adapted"]
            errors=[dict(id=a["id"],patient=a["patient"],criterion=a["input"]["criterion"],type=a["input"]["type"],
                         sentences=sentences(a["input"]["note"]),gold=a["target"],prediction=b,
                         label_error=a["target"]["label"]!=b["label"],
                         missing_evidence=sorted(set(a["target"]["evidence"])-set(b["evidence"])),
                         spurious_evidence=sorted(set(b["evidence"])-set(a["target"]["evidence"])))
                    for a,b in zip(rows,pred) if a["target"]["label"]!=b["label"] or set(a["target"]["evidence"])!=set(b["evidence"])]
            save("errors",errors)
    save("metrics_summary",summary)
    save("test_metrics_summary",[s for s in summary if s["split"]=="test"])


if __name__=="__main__":
    {"calibrate":calibrate,"final":final,"baselines":baselines}[sys.argv[1] if len(sys.argv)>1 else "final"]()
