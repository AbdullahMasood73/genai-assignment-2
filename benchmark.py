import json, time, statistics, threading, os
import psutil
import torch
from prepare import sentences
from model import Model, ROOT, choices, encode, greedy, TOK
from inference import reference,tiled,speculative
from train import read,save

def loaded():
    model=Model()
    missing,unexpected=model.load_state_dict(torch.load(ROOT/"results/adaptation.pt",weights_only=True),strict=False)
    assert missing==["embed_in.weight"] and not unexpected
    return model.eval()

def measure(fn,n=5):
    fn()
    elapsed=[]
    rss=[]
    process=psutil.Process()
    for _ in range(n):
        stop=threading.Event()
        def sample():
            while not stop.is_set():
                rss.append(process.memory_info().rss)
                stop.wait(.002)
        worker=threading.Thread(target=sample)
        worker.start()
        begin=time.perf_counter()
        try: value=fn()
        finally:
            elapsed.append(time.perf_counter()-begin)
            stop.set()
            worker.join()
            rss.append(process.memory_info().rss)
    return value,statistics.median(elapsed),max(rss)/2**20

@torch.no_grad()
def benchmarks():
    target=loaded()
    draft=Model(2).eval()
    torch.manual_seed(42)
    cases=[]
    # Use the same legal output vocabulary with EOS suppressed to control output length.
    for length,budget in [(32,4),(96,8),(192,16)]:
        prompt=encode(read("validation")[0])[:length]
        assert len(prompt)==length
        allowed=lambda out: [0,1,2,3,4,5,7,8,9]
        expected=greedy(target,prompt,allowed,False,budget)
        for cached in [False,True]:
            prefix=torch.tensor([prompt])
            def prefill(): return target(prefix)
            (logits,cache),pref,mem=measure(prefill)
            def decode():
                seq=list(prompt)
                output=[]
                scores=logits
                state=cache
                for i in range(budget):
                    a=allowed(output)
                    s=a[int(scores[0,-1,a].argmax())]
                    output.append(s)
                    if i+1<budget:
                        seq.append(TOK[s])
                        scores,state=target(torch.tensor([seq[-1:] if cached else seq]),state if cached else None)
                return output
            output,dec,peak=measure(decode)
            assert output==expected
            cases.append(dict(input=length,output=budget,mode="cached" if cached else "uncached",prefill_s=pref,decode_s=dec,tokens_s=budget/(pref+dec),peak_rss_mib=max(mem,peak),equal=True))
        (output,accepted,proposed),total,peak=measure(lambda: speculative(target,draft,prompt,allowed,budget))
        assert output==expected
        cases.append(dict(input=length,output=budget,mode="speculative",end_to_end_s=total,tokens_s=budget/total,peak_rss_mib=peak,accepted=accepted,proposed=proposed,equal=True))
    save("decoding",cases)
    rows=[]
    for length in [17,65,256]:
        q,k,v=torch.randn(3,1,4,length,32)
        for causal in [True,False]:
            a,ta,ma=measure(lambda:reference(q,k,v,causal))
            b,tb,mb=measure(lambda:tiled(q,k,v,causal))
            torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
            rows.append(dict(length=length,causal=causal,max_error=(a-b).abs().max().item(),mean_error=(a-b).abs().mean().item(),reference_s=ta,tiled_s=tb,reference_rss_mib=ma,tiled_rss_mib=mb))
    save("attention",rows)

if __name__=="__main__": benchmarks()
