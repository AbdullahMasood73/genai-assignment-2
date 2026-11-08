"""CPU implementation of the released Pythia-14M GPT-NeoX forward pass."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
TOKENIZER = Tokenizer.from_file(str(ROOT / "base/tokenizer.json"))
LABELS = ["included", "not included", "excluded", "not excluded", "not enough information", "not applicable"]
SYMBOLS = [" "+c for c in "ABCDEF"] + [";"] + [" "+str(i) for i in range(100)]
TOK = [TOKENIZER.encode(s).ids[0] for s in SYMBOLS]
assert all(len(TOKENIZER.encode(s).ids)==1 for s in SYMBOLS) and len(set(TOK))==len(TOK)
END = 6

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(128)
        self.post_attention_layernorm = nn.LayerNorm(128)
        self.attention = nn.ModuleDict({"query_key_value":nn.Linear(128,384),"dense":nn.Linear(128,128)})
        self.mlp = nn.ModuleDict({"dense_h_to_4h":nn.Linear(128,512),"dense_4h_to_h":nn.Linear(512,128)})

    def forward(self, x, past):
        length = x.shape[1]
        start = 0 if past is None else past[0].shape[2]
        pos = torch.arange(start,start+length,device=x.device)
        freq = pos[:,None] * (10000 ** (-torch.arange(0,8,2,device=x.device)/8))
        angle = torch.cat([freq,freq],-1)
        def rotate(t):
            a,b = t[...,:8].chunk(2,-1)
            r = t[...,:8]*angle.cos() + torch.cat([-b,a],-1)*angle.sin()
            return torch.cat([r,t[...,8:]],-1)
        q,k,v = self.attention["query_key_value"](self.input_layernorm(x)).view(x.shape[0],length,4,96).transpose(1,2).chunk(3,-1)
        q,k = rotate(q),rotate(k)
        if past is not None:
            k,v = torch.cat([past[0],k],2),torch.cat([past[1],v],2)
        mask = None if start==0 else pos[:,None]>=torch.arange(k.shape[2],device=x.device)
        a = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=start==0)
        h = self.mlp["dense_h_to_4h"](self.post_attention_layernorm(x))
        return x+self.attention["dense"](a.transpose(1,2).reshape_as(x))+self.mlp["dense_4h_to_h"](F.gelu(h)),(k,v)

class Model(nn.Module):
    def __init__(self, layers=6):
        super().__init__()
        state=load_file(str(ROOT/"base/model.safetensors"))
        self.embed_in=nn.Embedding.from_pretrained(state["gpt_neox.embed_in.weight"].float())
        self.layers=nn.ModuleList([Block() for _ in range(layers)])
        self.final_layer_norm=nn.LayerNorm(128)
        self.head=nn.Linear(128,len(TOK),bias=False)
        with torch.no_grad():
            self.head.weight.copy_(state["embed_out.weight"][TOK])
        for name,module in [("final_layer_norm",self.final_layer_norm)]+[(f"layers.{i}",layer) for i,layer in enumerate(self.layers)]:
            prefix="gpt_neox."+name+"."
            module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
    def forward(self, ids, cache=None):
        x=self.embed_in(ids)
        updated=[]
        for i,layer in enumerate(self.layers):
            x,kv=layer(x,None if cache is None else cache[i])
            updated.append(kv)
        return self.head(self.final_layer_norm(x)),updated

def encode(item):
    raw=item["input"]
    ids=TOKENIZER.encode("Codes: A included, B not included, C excluded, D not excluded, E not enough information, F not applicable. Give code then evidence sentence numbers;\n"+raw["type"]+" criterion: "+(raw["criterion"] or "[criterion text missing]")+"\nPatient:\n"+raw["note"]+"\nAnswer:").ids
    if len(ids)>1930:
        raise ValueError("Input exceeds the supported context. Shorten the note or criterion.")
    return ids

def choices(kind, sentence_ids, generated):
    if not generated:
        return ([0,1] if kind=="inclusion" else [2,3])+[4,5]
    return [END]+[7+i for i in sentence_ids if len(generated)==1 or 7+i>generated[-1]]

@torch.no_grad()
def greedy(model, ids, allowed, cached=True, limit=102):
    seq=list(ids)
    output=[]
    cache=None
    for _ in range(limit):
        chunk=seq[-1:] if cached and cache is not None else seq
        logits,new=model(torch.tensor([chunk]),cache if cached else None)
        cache=new
        valid=allowed(output)
        symbol=valid[int(logits[0,-1,valid].argmax())]
        output.append(symbol)
        if symbol==END: break
        seq.append(TOK[symbol])
    return output


