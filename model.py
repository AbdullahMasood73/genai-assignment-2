"""From-scratch CPU implementation of the released Pythia (GPT-NeoX) forward pass.

The architecture is read from ``<base>/config.json`` so the same code runs the
Pythia-14M and Pythia-70M checkpoints. Only the input embedding is taken as a
frozen lookup table; every decoder block, the final LayerNorm and a restricted
output head are ordinary trainable ``nn.Module`` parameters.
"""
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
# All Pythia sizes share one byte-level BPE tokenizer (vocab padded to 50,304).
TOKENIZER = Tokenizer.from_file(str(ROOT / "base/tokenizer.json"))
LABELS = ["included", "not included", "excluded", "not excluded", "not enough information", "not applicable"]
# Output alphabet: six label codes, a stop symbol and sentence IDs 0..99.
SYMBOLS = [" "+c for c in "ABCDEF"] + [";"] + [" "+str(i) for i in range(100)]
TOK = [TOKENIZER.encode(s).ids[0] for s in SYMBOLS]
assert all(len(TOKENIZER.encode(s).ids)==1 for s in SYMBOLS) and len(set(TOK))==len(TOK)
END = 6
MAX_PROMPT = 1930      # 1930 prompt + 102 generated positions < 2048 context


def config(base="base"):
    return json.loads((ROOT/base/"config.json").read_text())


class Block(nn.Module):
    """One GPT-NeoX block: parallel attention + MLP residual, partial rotary embedding."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg["hidden_size"]
        self.heads = cfg["num_attention_heads"]
        self.head_dim = d // self.heads
        self.rot = int(self.head_dim * cfg["rotary_pct"])
        self.base = cfg["rotary_emb_base"]
        self.input_layernorm = nn.LayerNorm(d, eps=cfg["layer_norm_eps"])
        self.post_attention_layernorm = nn.LayerNorm(d, eps=cfg["layer_norm_eps"])
        self.attention = nn.ModuleDict({"query_key_value":nn.Linear(d,3*d),"dense":nn.Linear(d,d)})
        self.mlp = nn.ModuleDict({"dense_h_to_4h":nn.Linear(d,cfg["intermediate_size"]),
                                  "dense_4h_to_h":nn.Linear(cfg["intermediate_size"],d)})

    def forward(self, x, past):
        b, length, _ = x.shape
        # Absolute positions continue from the cached length (0 on prefill).
        start = 0 if past is None else past[0].shape[2]
        pos = torch.arange(start, start+length, device=x.device)
        freq = pos[:,None] * (self.base ** (-torch.arange(0,self.rot,2,device=x.device)/self.rot))
        angle = torch.cat([freq,freq],-1)
        cos, sin = angle.cos(), angle.sin()
        r = self.rot
        def rotate(t):
            a,c = t[...,:r].chunk(2,-1)
            return torch.cat([t[...,:r]*cos + torch.cat([-c,a],-1)*sin, t[...,r:]],-1)
        # GPT-NeoX packs q,k,v per head: [head, 3*head_dim].
        q,k,v = self.attention["query_key_value"](self.input_layernorm(x)).view(b,length,self.heads,3*self.head_dim).transpose(1,2).chunk(3,-1)
        q,k = rotate(q),rotate(k)          # rotary applied before caching
        if past is not None:
            k,v = torch.cat([past[0],k],2),torch.cat([past[1],v],2)
        # Query at absolute position p may attend to keys 0..p.
        mask = None if start==0 else pos[:,None]>=torch.arange(k.shape[2],device=x.device)
        a = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=start==0 and length>1)
        h = self.mlp["dense_h_to_4h"](self.post_attention_layernorm(x))
        return x+self.attention["dense"](a.transpose(1,2).reshape(b,length,-1))+self.mlp["dense_4h_to_h"](F.gelu(h)),(k,v)


class Model(nn.Module):
    def __init__(self, layers=None, base="base"):
        super().__init__()
        cfg = config(base)
        self.cfg = cfg
        state=load_file(str(ROOT/base/"model.safetensors"))
        d = cfg["hidden_size"]
        self.embed_in=nn.Embedding.from_pretrained(state["gpt_neox.embed_in.weight"].float())
        self.layers=nn.ModuleList([Block(cfg) for _ in range(layers or cfg["num_hidden_layers"])])
        self.final_layer_norm=nn.LayerNorm(d, eps=cfg["layer_norm_eps"])
        # Restricted head: the pretrained unembedding rows of the 107 legal symbols.
        self.head=nn.Linear(d,len(TOK),bias=False)
        with torch.no_grad():
            self.head.weight.copy_(state["embed_out.weight"][TOK].float())
        for name,module in [("final_layer_norm",self.final_layer_norm)]+[(f"layers.{i}",layer) for i,layer in enumerate(self.layers)]:
            prefix="gpt_neox."+name+"."
            module.load_state_dict({k[len(prefix):]:v.float() for k,v in state.items()
                                    if k.startswith(prefix) and not k.endswith(("attention.bias","attention.masked_bias","rotary_emb.inv_freq"))})

    def forward(self, ids, cache=None):
        x=self.embed_in(ids)
        updated=[]
        for i,layer in enumerate(self.layers):
            x,kv=layer(x,None if cache is None else cache[i])
            updated.append(kv)
        return self.head(self.final_layer_norm(x)),updated


def deployment():
    """Frozen target/draft/decoding choice shared by evaluation, benchmarks and the app."""
    return json.loads((ROOT/"results/deployment.json").read_text())


def load_adapted(spec):
    """Build a model from a deployment entry {base, checkpoint, layers?} and load trained weights."""
    model=Model(spec.get("layers"),spec["base"])
    if spec.get("checkpoint"):
        state=torch.load(ROOT/spec["checkpoint"],weights_only=True)
        n=len(model.layers)
        # An early-exit draft keeps only the first n trained blocks of the target.
        state={k:v for k,v in state.items() if not k.startswith("layers.") or int(k.split(".")[1])<n}
        missing,unexpected=model.load_state_dict(state,strict=False)
        assert missing==["embed_in.weight"] and not unexpected, (missing,unexpected)
    return model.eval()


def kv_cache_bytes(model, length):
    """Analytic FP32 K/V cache size: 2 tensors x layers x hidden x positions x 4 bytes."""
    return 2*len(model.layers)*model.cfg["hidden_size"]*length*4


INSTRUCTION = ("Codes: A included, B not included, C excluded, D not excluded, E not enough information, "
               "F not applicable. Give code then evidence sentence numbers;\n")


def encode(item):
    raw=item["input"]
    ids=TOKENIZER.encode(INSTRUCTION+raw["type"]+" criterion: "+(raw["criterion"] or "[criterion text missing]")
                         +"\nPatient:\n"+raw["note"]+"\nAnswer:").ids
    if len(ids)>MAX_PROMPT:
        raise ValueError(f"Input is {len(ids)} tokens; the supported limit is {MAX_PROMPT}. Shorten the note or criterion.")
    return ids


def choices(kind, sentence_ids, generated):
    """Grammar: one type-specific label, then strictly ascending sentence IDs, then stop."""
    if not generated:
        return ([0,1] if kind=="inclusion" else [2,3])+[4,5]
    return [END]+[7+i for i in sentence_ids if len(generated)==1 or 7+i>generated[-1]]


def biased(logits, valid, end_bias=0.0):
    """Restrict logits to legal symbols and add a constant offset to the stop symbol."""
    scores = logits[valid].clone()
    if end_bias and END in valid:
        scores[valid.index(END)] += end_bias
    return scores


@torch.no_grad()
def greedy(model, ids, allowed, cached=True, limit=102, end_bias=0.0):
    """Greedy constrained decoding.

    cached=True : prefill once, then feed only the newest token with the per-layer
                  (K,V) cache; positions continue from the cache length.
    cached=False: reference path that re-encodes the whole sequence every step.
    """
    seq=list(ids)
    output=[]
    cache=None
    for _ in range(limit):
        chunk=seq[-1:] if cached and cache is not None else seq
        logits,new=model(torch.tensor([chunk]),cache if cached else None)
        cache=new
        valid=allowed(output)
        symbol=valid[int(biased(logits[0,-1],valid,end_bias).argmax())]
        output.append(symbol)
        if symbol==END: break
        seq.append(TOK[symbol])
    return output
