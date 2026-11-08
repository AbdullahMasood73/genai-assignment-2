"""Greedy speculative decoding and educational attention kernels.

Speculative invariant: before each round, both caches hold exactly the committed
prefix *excluding* the pending token (the last committed token, not yet fed).
"""
import torch
from model import TOK, END, biased


def trim(cache, length):
    """Roll a per-layer (K,V) cache back to its first ``length`` positions."""
    return [(k[:,:,:length],v[:,:,:length]) for k,v in cache]


def select(logits, permitted, end_bias=0.0):
    return permitted[int(biased(logits,permitted,end_bias).argmax())]


@torch.no_grad()
def speculative(target,draft,ids,allowed,limit=102,width=4,end_bias=0.0):
    """Greedy speculative decoding; returns (tokens, accepted, proposed).

    1. Draft proposes up to ``width`` symbols autoregressively with its own cache.
    2. One target forward over [pending]+proposals scores every position at once.
    3. The longest prefix where target argmax == proposal is accepted; at the first
       mismatch the target's own token is emitted (rejection correction) and both
       caches are trimmed back to the accepted prefix.
    4. If all proposals match, the target's prediction after the last proposal is a
       free "bonus" token.
    The output is therefore identical to target-only greedy decoding.
    """
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
            s=select(logits[0,-1],allowed(out+guesses),end_bias)
            guesses.append(s)
            token=TOK[s]
            if s==END: break
        proposed+=len(guesses)
        logits,tc=target(torch.tensor([[pending]+[TOK[s] for s in guesses]]),tc)
        for j,s in enumerate(guesses):
            correct=select(logits[0,j],allowed(out),end_bias)
            out.append(correct)
            accepted+=int(correct==s)
            if correct==END or len(out)==limit: return out,accepted,proposed
            if correct!=s:
                # Cache keeps prefix + pending + accepted guesses[:j]; the corrected
                # token becomes the new pending token.
                tc,dc=trim(tc,start+j+1),trim(dc,start+j+1)
                pending=TOK[correct]
                break
        else:
            # All proposals accepted: the draft has not yet consumed its last guess.
            _,dc=draft(torch.tensor([[TOK[guesses[-1]]]]),dc)
            bonus=select(logits[0,-1],allowed(out),end_bias)
            out.append(bonus)
            pending=TOK[bonus]
            if bonus==END: break
    return out,accepted,proposed


def reference(q,k,v,causal=True):
    """Materialised scaled dot-product attention with a stable (max-shifted) softmax."""
    scores=q@k.transpose(-2,-1)*q.shape[-1]**-0.5
    if causal:
        # Query i (aligned to the end of the key sequence) may see keys <= i + offset.
        offset=k.size(-2)-q.size(-2)
        scores=scores.masked_fill(torch.ones_like(scores,dtype=torch.bool).triu(1+offset),-torch.inf)
    scores=scores-scores.amax(-1,keepdim=True)
    weights=scores.exp()
    return (weights/weights.sum(-1,keepdim=True))@v


def tiled(q,k,v,causal=True,block=32):
    """Blocked attention with an online softmax (FlashAttention-style recurrence).

    For each query tile keep a running row maximum m, normaliser z and numerator acc.
    For each key tile with scores S: m' = max(m, rowmax S); P = exp(S - m');
    z = e^(m-m') z + rowsum P; acc = e^(m-m') acc + P V. Output = acc / z.
    Only a block x block score tile is ever materialised.
    """
    offset=k.size(-2)-q.size(-2)
    result=torch.empty(*q.shape[:-1],v.size(-1),dtype=v.dtype)
    for i in range(0,q.size(-2),block):
        qb=q[...,i:i+block,:]
        m=torch.full_like(qb[...,:1],-torch.inf)
        z=torch.zeros_like(m)
        acc=torch.zeros(*qb.shape[:-1],v.size(-1),dtype=v.dtype)
        rows=torch.arange(i,i+qb.size(-2))+offset
        for j in range(0,k.size(-2),block):
            if causal and j>rows[-1]: break          # wholly future key tile
            kb=k[...,j:j+block,:]
            scores=qb@kb.transpose(-2,-1)*q.shape[-1]**-0.5
            if causal and j+kb.size(-2)-1>rows[0]:  # tile straddles the diagonal
                mask=torch.arange(j,j+kb.size(-2))[None,:]>rows[:,None]
                scores=scores.masked_fill(mask,-torch.inf)
            new=torch.maximum(m,scores.amax(-1,keepdim=True))
            scale=(m-new).exp()
            weights=(scores-new).exp()
            z=z*scale+weights.sum(-1,keepdim=True)
            acc=acc*scale+weights@v[...,j:j+block,:]
            m=new
        result[...,i:i+block,:]=acc/z
    return result
