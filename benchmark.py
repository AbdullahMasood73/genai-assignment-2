"""Controlled decoding and attention benchmarks (offline logs and the app's live view).

Timing protocol: one warm-up call, then the median of ``repeats`` synchronous calls
(CPU kernels are synchronous; CUDA would be synchronised explicitly). Peak memory
is the maximum process RSS sampled every 2 ms during the timed calls.
"""
import json, time, statistics, threading, platform, sys
import psutil
import torch
from torch.nn import functional as F
from model import ROOT, TOK, END, encode, deployment, load_adapted, kv_cache_bytes, biased, choices
from inference import reference, tiled, speculative
from prepare import sentences

# Controlled workloads use a fixed nine-symbol alphabet with the stop symbol
# suppressed, so every path generates exactly the requested number of tokens.
FIXED=lambda out: [0,1,2,3,4,5,7,8,9]


def loaded():
    return load_adapted(deployment()["target"])


def draft_model(kind=None):
    d=deployment()
    return load_adapted(d["drafts"][kind or d["draft"]])


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def measure(fn,repeats=5):
    """Warm-up once, then return (last value, median seconds, all seconds, peak RSS MiB)."""
    fn()
    elapsed,rss=[],[]
    process=psutil.Process()
    for _ in range(repeats):
        stop=threading.Event()
        def sample():
            while not stop.is_set():
                rss.append(process.memory_info().rss)
                stop.wait(.002)
        worker=threading.Thread(target=sample)
        worker.start()
        sync(); begin=time.perf_counter()
        try:
            value=fn(); sync()
        finally:
            elapsed.append(time.perf_counter()-begin)
            stop.set(); worker.join()
            rss.append(process.memory_info().rss)
    return value,statistics.median(elapsed),elapsed,max(rss)/2**20


@torch.no_grad()
def decode_paths(target,draft,prompt,budget,allowed=FIXED,repeats=5,width=4,end_bias=0.0):
    """Benchmark uncached, cached and speculative greedy decoding on one prompt."""
    rows=[]
    prefix=torch.tensor([prompt])
    outputs={}
    margins=[]
    for cached in [False,True]:
        (logits,cache),pref,_,mem=measure(lambda: target(prefix),repeats)
        def decode():
            seq=list(prompt); output=[]; scores=logits; state=cache
            for i in range(budget):
                a=allowed(output)
                b=biased(scores[0,-1],a,end_bias)
                s=a[int(b.argmax())]
                if cached and len(a)>1:
                    top=b.topk(2).values; margins.append((top[0]-top[1]).item())
                output.append(s)
                if s==END: break
                if i+1<budget:
                    seq.append(TOK[s])
                    # Cached: feed one token and reuse K/V. Uncached: re-encode everything.
                    scores,state=target(torch.tensor([seq[-1:] if cached else seq]),state if cached else None)
            return output
        output,dec,_,peak=measure(decode,repeats)
        mode="cached" if cached else "uncached"
        outputs[mode]=output
        rows.append(dict(input=len(prompt),output=len(output),mode=mode,prefill_ms=1000*pref,decode_ms=1000*dec,
                         total_ms=1000*(pref+dec),tokens_s=len(output)/(pref+dec),peak_rss_mib=max(mem,peak),
                         kv_cache_kib=kv_cache_bytes(target,len(prompt)+len(output))/1024 if cached else 0))
    (output,accepted,proposed),total,_,peak=measure(lambda: speculative(target,draft,prompt,allowed,budget,width,end_bias),repeats)
    outputs["speculative"]=output
    rows.append(dict(input=len(prompt),output=len(output),mode="speculative",prefill_ms=None,decode_ms=None,total_ms=1000*total,
                     tokens_s=len(output)/total,peak_rss_mib=peak,accepted=accepted,proposed=proposed,
                     acceptance=accepted/proposed if proposed else None,kv_cache_kib=None))
    ref=outputs["uncached"]
    for r in rows:
        o=outputs[r["mode"]]
        r["equal"]=o==ref
        r["first_mismatch"]=next((i for i,(x,y) in enumerate(zip(o,ref)) if x!=y),None if len(o)==len(ref) else min(len(o),len(ref)))
    cached_ms=rows[1]["total_ms"]
    for r in rows: r["speedup_vs_uncached"]=rows[0]["total_ms"]/r["total_ms"]; r["speedup_vs_cached"]=cached_ms/r["total_ms"]
    return rows,(min(margins) if margins else None)


def long_prompt(length):
    """Concatenate validation prompts (never test) until the requested length."""
    from train import read
    ids=[]
    for row in read("validation"):
        ids+=encode(row)
        if len(ids)>=length: return ids[:length]
    raise ValueError("Not enough validation text")


@torch.no_grad()
def attention_check(length_q,length_k,causal,block=32,heads=4,dim=32,repeats=5,seed=0):
    g=torch.Generator().manual_seed(seed)
    q=torch.randn(1,heads,length_q,dim,generator=g); k=torch.randn(1,heads,length_k,dim,generator=g); v=torch.randn(1,heads,length_k,dim,generator=g)
    mask=torch.ones(length_q,length_k,dtype=torch.bool).tril(length_k-length_q) if causal else None
    truth=F.scaled_dot_product_attention(q.double(),k.double(),v.double(),attn_mask=mask)
    a,ta,_,ma=measure(lambda:reference(q,k,v,causal),repeats)
    b,tb,_,mb=measure(lambda:tiled(q,k,v,causal,block),repeats)
    c,tc,_,mc=measure(lambda:F.scaled_dot_product_attention(q,k,v,attn_mask=mask),repeats)
    err=lambda x:(x.double()-truth).abs()
    return dict(q_len=length_q,k_len=length_k,causal=causal,block=block,
                ref_max_error=err(a).max().item(),ref_mean_error=err(a).mean().item(),
                tiled_max_error=err(b).max().item(),tiled_mean_error=err(b).mean().item(),
                tiled_vs_ref_max=(a-b).abs().max().item(),
                reference_ms=1000*ta,tiled_ms=1000*tb,sdpa_ms=1000*tc,
                reference_rss_mib=ma,tiled_rss_mib=mb,sdpa_rss_mib=mc,
                full_scores_kib=heads*length_q*length_k*4/1024,tile_scores_kib=heads*min(block,length_q)*min(block,length_k)*4/1024)


def stability_check():
    """Logits of magnitude ~1e4 overflow a naive exp; both stable kernels stay finite."""
    q=torch.full((1,1,8,4),60.); k=torch.full((1,1,8,4),60.); v=torch.randn(1,1,8,4)
    naive=(q@k.transpose(-2,-1)*0.5).exp()
    out=[reference(q,k,v,True),tiled(q,k,v,True,block=3)]
    return dict(naive_has_inf=bool(torch.isinf(naive).any()),reference_finite=bool(torch.isfinite(out[0]).all()),
                tiled_finite=bool(torch.isfinite(out[1]).all()),max_difference=(out[0]-out[1]).abs().max().item())


def environment():
    return dict(python=sys.version.split()[0],torch=torch.__version__,threads=torch.get_num_threads(),
                processor=platform.processor(),ram_gib=round(psutil.virtual_memory().total/2**30,1))


@torch.no_grad()
def benchmarks():
    from train import read
    dep=deployment()
    target=loaded()
    torch.manual_seed(42)
    log=dict(environment=environment(),deployment=dep,fixed_length=[],screening=[],attention=[])
    drafts={name:load_adapted(spec) for name,spec in dep["drafts"].items()}
    # 1. Controlled lengths (stop suppressed) for every draft choice.
    for length,budget in [(32,8),(256,32),(1024,64)]:
        prompt=long_prompt(length)
        for name,draft in drafts.items():
            rows,margin=decode_paths(target,draft,prompt,budget)
            for r in rows: r["draft"]=name; r["min_top2_margin"]=margin
            log["fixed_length"]+=rows if name==dep["draft"] else [r for r in rows if r["mode"]=="speculative"]
            assert all(r["equal"] for r in rows),rows
            print(length,budget,name,[(r["mode"],round(r["total_ms"],1)) for r in rows],flush=True)
    # 2. Real screening outputs (grammar + stop) on the first 24 validation records.
    for row in read("validation")[:24]:
        prompt=encode(row)
        ids=sorted(sentences(row["input"]["note"]))
        allowed=lambda out,k=row["input"]["type"],ids=ids: choices(k,ids,out)
        for name,draft in drafts.items():
            rows,_=decode_paths(target,draft,prompt,102,allowed,repeats=3,end_bias=dep["end_bias"])
            assert all(r["equal"] for r in rows)
            for r in rows: r["draft"]=name; r["id"]=row["id"]
            log["screening"]+=rows if name==dep["draft"] else [r for r in rows if r["mode"]=="speculative"]
    # 3. Attention kernels: square causal/non-causal, rectangular (cached decode) shapes.
    for lq,lk in [(17,17),(65,65),(256,256),(512,512),(1,300),(40,300)]:
        for causal in [True,False]:
            log["attention"].append(attention_check(lq,lk,causal))
    for block in [8,16,64,128]:
        log["attention"].append(attention_check(256,256,True,block))
    log["stability"]=stability_check()
    (ROOT/"results/benchmark_log.json").write_text(json.dumps(log,indent=2))
    print("saved results/benchmark_log.json")


if __name__=="__main__": benchmarks()
