"""Independent acceptance tests. Run: python verify.py  (prints one line per check group)."""
import json, hashlib
import numpy as np
import torch
from torch.nn import functional as F
import pyarrow.parquet as pq
from model import ROOT, Model, greedy, encode, TOK, END, choices, deployment, load_adapted, LABELS
from inference import speculative, trim, reference, tiled
from train import read
from prepare import sentences, SOURCE
import app

results=[]
def ok(name): results.append(name); print("PASS", name, flush=True)


def check_split():
    rows=pq.read_table(SOURCE).to_pylist()
    manifest=json.loads((ROOT/"data/manifest.json").read_text())
    assert manifest["source_sha256"]==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    ids=sorted({r["patient_id"] for r in rows}); np.random.default_rng(42).shuffle(ids)
    expect={"train":ids[:37],"validation":ids[37:45],"test":ids[45:]}
    seen=set(); total=0
    for name,patients in expect.items():
        assert manifest["splits"][name]["patients"]==patients
        data=read(name)
        assert {r["patient"] for r in data}==set(patients) and len(data)==manifest["splits"][name]["records"]
        assert not seen&set(patients); seen|=set(patients); total+=len(data)
        for r in data:
            assert set(r["input"])=={"note","type","criterion"}      # no GPT-4 predictions/explanations
    assert total==len(rows)==sum(1 for r in rows) and len(seen)==53
    leaked=[c for c in rows[0] if c.startswith("gpt") or "pred" in c.lower()]
    ok(f"split 37/8/8 re-derived from parquet, patient-disjoint, {total} rows; inputs allowlisted (excluded columns: {leaked})")


@torch.no_grad()
def check_cache(target):
    prompts=[encode(r) for r in read("validation")[:3]]
    worst=0
    for prompt in prompts:
        x=torch.tensor([prompt])
        full,_=target(x)
        _,cache=target(x[:,:-4])
        chunk,updated=target(x[:,-4:],cache)
        worst=max(worst,(chunk-full[:,-4:]).abs().max().item())
        assert torch.equal(chunk.argmax(-1),full[:,-4:].argmax(-1))
        assert all(k.shape[2]==x.shape[1] for k,v in updated)
        # Rollback: trimming a longer cache equals a fresh prefill of the prefix.
        _,short=target(x[:,:-4]); trimmed=trim(updated,x.shape[1]-4)
        assert all(torch.allclose(a[0],b[0],atol=1e-5) for a,b in zip(short,trimmed))
    ok(f"chunked cached logits match full recompute (max |diff| {worst:.1e}); cache grows by fed length; trim == fresh prefix")


@torch.no_grad()
def check_greedy(target):
    n=0
    for row in read("validation")[:12]:
        ids=sorted(sentences(row["input"]["note"]))
        allowed=lambda out,k=row["input"]["type"]: choices(k,ids,out)
        prompt=encode(row)
        assert greedy(target,prompt,allowed,True)==greedy(target,prompt,allowed,False); n+=1
    fixed=lambda out:[0,1,2,3,4,5,7,8,9]
    for length,budget in [(16,4),(64,16),(200,32),(400,48)]:
        prompt=encode(read("validation")[1])[:length] if length<=300 else (encode(read("validation")[1])+encode(read("validation")[2]))[:length]
        assert greedy(target,prompt,fixed,True,budget)==greedy(target,prompt,fixed,False,budget); n+=1
    ok(f"cached greedy == uncached greedy on {n} prompts (12 real screening outputs + 4 controlled lengths)")


@torch.no_grad()
def check_speculative(target,drafts,dep):
    prompts=[encode(r) for r in read("validation")[:3]]
    for name,draft in drafts.items():
        for budget in [1,2,4,5,9]:            # all-accept, partial final block, budget stop
            permitted=lambda out:[7+len(out)%3]
            reference_out=greedy(target,prompts[0],permitted,True,budget)
            actual,accepted,proposed=speculative(target,draft,prompts[0],permitted,budget)
            assert actual==reference_out and accepted==proposed
        assert speculative(target,draft,prompts[0],lambda out:[0] if not out else [END])[0]==[0,END]   # early stop
        fixed=lambda out:[0,1,2,3,4,5,7,8,9]
        for p in prompts:
            for width in [1,3,4,8]:
                assert speculative(target,draft,p,fixed,24,width)[0]==greedy(target,p,fixed,True,24)
        for row in read("validation")[:10]:
            ids=sorted(sentences(row["input"]["note"]))
            allowed=lambda out,k=row["input"]["type"]: choices(k,ids,out)
            p=encode(row)
            assert speculative(target,draft,p,allowed,end_bias=dep["end_bias"])[0]==greedy(target,p,allowed,end_bias=dep["end_bias"])
    ok(f"speculative == target greedy for drafts {list(drafts)} (forced accept/reject/stop, widths 1-8, real outputs)")


def check_attention():
    torch.manual_seed(0); worst=0
    for lq,lk in [(1,1),(7,7),(17,17),(33,33),(65,65),(1,50),(20,90),(128,128)]:
        for causal in [True,False]:
            for block in [4,16,32]:
                q,k,v=torch.randn(1,4,lq,32),torch.randn(1,4,lk,32),torch.randn(1,4,lk,32)
                mask=torch.ones(lq,lk,dtype=torch.bool).tril(lk-lq) if causal else None
                truth=F.scaled_dot_product_attention(q.double(),k.double(),v.double(),attn_mask=mask)
                for fn in (lambda:reference(q,k,v,causal),lambda:tiled(q,k,v,causal,block)):
                    e=(fn().double()-truth).abs().max().item(); worst=max(worst,e); assert e<1e-5
    q=torch.full((1,1,8,4),60.); k=q.clone(); v=torch.randn(1,1,8,4)
    assert torch.isfinite(reference(q,k,v)).all() and torch.isfinite(tiled(q,k,v,True,3)).all()
    ok(f"reference and tiled attention match FP64 SDPA on 48 configurations (max |err| {worst:.1e}); stable for large logits")


def check_app(target,draft,dep):
    r=app.screen({"note":"8. A person reports a cough.\n\n12. The person is 38 years old.","criteria":["inclusion | Adult.","exclusion | Pregnancy."],"mode":"compare"},target,draft,dep)
    assert all(x["agreement"] for x in r["results"]) and r["numbering"]=="original numbering"
    assert all(e["id"] in {8,12} for x in r["results"] for e in x["evidence"])
    assert r["rule"] in ("R1","R2","R3") and "not a clinical enrollment decision" in r["notice"]
    r=app.screen({"note":"A 40 year old man. He smokes daily.","criteria":"inclusion | Age over 18","mode":"speculative"},target,draft,dep)
    assert [s["id"] for s in r["sentences"]]==[0,1]
    bad=[{},[],{"note":"Test."},{"note":"Test.","criteria":["inclusion |"]},{"note":"Test.","criteria":["rule"]},
         {"note":"  ","criteria":["inclusion | x"]},{"note":"1. a\n1. b","criteria":["inclusion | x"]},
         {"note":"Test.","criteria":["inclusion | x"],"mode":"fast"},{"note":"x "*5000,"criteria":["inclusion | x"]},
         {"note":"Test.","criteria":["inclusion | x"]*31},{"note":5,"criteria":["inclusion | x"]}]
    for payload in bad:
        try: app.screen(payload,target,draft,dep)
        except ValueError: continue
        raise AssertionError(f"Invalid request accepted: {str(payload)[:80]}")
    rows=[dict(label="included",evidence=[1],unsupported=False,criterion="a"),dict(label="not enough information",evidence=[],unsupported=False,criterion="b")]
    assert app.aggregate(rows)["rule"]=="R2"
    rows.append(dict(label="excluded",evidence=[],unsupported=True,criterion="c"))
    s=app.aggregate(rows); assert s["rule"]=="R1" and s["unsupported"]==["c"]
    for payload in [{"prompt_tokens":4},{"output_tokens":0},{"repeats":"x"}]:
        try: app.live_benchmark(payload,target,draft,{dep["draft"]:draft},dep)
        except ValueError: continue
        raise AssertionError("Invalid benchmark accepted")
    ok(f"app: original IDs kept, blank lines ignored, three-mode agreement, rules R1-R5, {len(bad)+3} invalid requests rejected")


if __name__=="__main__":
    dep=deployment()
    target=load_adapted(dep["target"])
    drafts={k:load_adapted(v) for k,v in dep["drafts"].items()}
    check_split(); check_cache(target); check_greedy(target); check_speculative(target,drafts,dep)
    check_attention(); check_app(target,drafts[dep["draft"]],dep)
    (ROOT/"results/verify_log.json").write_text(json.dumps(results,indent=2))
    print(f"All {len(results)} check groups passed.")
