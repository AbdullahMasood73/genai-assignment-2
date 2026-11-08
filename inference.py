"""Greedy block verification. Caches store the prefix before the pending token."""
import torch
from model import TOK, END

def trim(cache, length):
    return [(k[:,:,:length],v[:,:,:length]) for k,v in cache]

def select(logits, permitted):
    return permitted[int(logits[permitted].argmax())]

@torch.no_grad()
def speculative(target,draft,ids,allowed,limit=102,width=4):
    # Keep the final prompt token pending so each block returns its first prediction.
    _,tc=target(torch.tensor([ids[:-1]]))
    _,dc=draft(torch.tensor([ids[:-1]]))
    pending=ids[-1]
    out=[]
    accepted=proposed=0
    while len(out)<limit:
        start=tc[0][0].shape[2]
        guesses=[]
        token=pending
        for _ in range(min(width,limit-len(out))):
            logits,dc=draft(torch.tensor([[token]]),dc)
            s=select(logits[0,-1],allowed(out+guesses))
            guesses.append(s)
            token=TOK[s]
            if s==END: break
        proposed+=len(guesses)
        logits,tc=target(torch.tensor([[pending]+[TOK[s] for s in guesses]]),tc)
        for j,s in enumerate(guesses):
            correct=select(logits[0,j],allowed(out))
            out.append(correct)
            accepted+=int(correct==s)
            if correct==END or len(out)==limit: return out,accepted,proposed
            if correct!=s:
                tc,dc=trim(tc,start+j+1),trim(dc,start+j+1)
                pending=TOK[correct]
                break
        else:
            _,dc=draft(torch.tensor([[TOK[guesses[-1]]]]),dc)
            bonus=select(logits[0,-1],allowed(out))
            out.append(bonus)
            pending=TOK[bonus]
            if bonus==END: break
    return out,accepted,proposed

def reference(q,k,v,causal=True):
    scores=q@k.transpose(-2,-1)*q.shape[-1]**-0.5
    if causal: scores=scores.masked_fill(torch.ones_like(scores,dtype=torch.bool).triu(1),-torch.inf)
    return scores.softmax(-1)@v

def tiled(q,k,v,causal=True,block=32):
    result=torch.empty_like(v)
    for i in range(0,q.size(-2),block):
        qb=q[...,i:i+block,:]
        m=torch.full_like(qb[...,:1],-torch.inf)
        z=torch.zeros_like(m)
        acc=torch.zeros_like(qb)
        for j in range(0,k.size(-2),block):
            if causal and j>=i+qb.size(-2): break
            kb=k[...,j:j+block,:]
            scores=qb@kb.transpose(-2,-1)*q.shape[-1]**-0.5
            if causal:
                mask=torch.arange(j,j+kb.size(-2))[None,:]>torch.arange(i,i+qb.size(-2))[:,None]
                scores=scores.masked_fill(mask,-torch.inf)
            new=torch.maximum(m,scores.amax(-1,keepdim=True))
            scale=(m-new).exp()
            weights=(scores-new).exp()
            z=z*scale+weights.sum(-1,keepdim=True)
            acc=acc*scale+weights@v[...,j:j+block,:]
            m=new
        result[...,i:i+block,:]=acc/z
    return result

