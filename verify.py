"""Independent acceptance tests for numerical and application behavior."""
import json
import torch
from model import ROOT,Model,greedy,encode,TOK,END
from inference import speculative
from app import screen
from benchmark import loaded
from train import read
from prepare import sentences

def verify():
    target=loaded()
    draft=Model(2).eval()
    prompts=[encode(row) for row in read("validation")[:3]]
    cases=[]
    # Forced paths exercise accepted blocks, a partial final block, and early EOS.
    for budget in [1,2,4,5,9]:
        permitted=lambda out:[7+len(out)%3]
        reference=greedy(target,prompts[0],permitted,True,budget)
        actual,accepted,proposed=speculative(target,draft,prompts[0],permitted,budget)
        assert actual==reference and accepted==proposed
        cases.append(budget)
    allowed=lambda out:[0] if not out else [END]
    assert speculative(target,draft,prompts[0],allowed)[0]==[0,END]
    for prompt in prompts:
        x=torch.tensor([prompt])
        with torch.no_grad():
            full,_=target(x)
            _,cache=target(x[:,:-4])
            chunk,updated=target(x[:,-4:],cache)
            print("Cached logit max absolute difference:", (chunk-full[:,-4:]).abs().max().item())
            assert torch.equal(chunk.argmax(-1),full[:,-4:].argmax(-1))
            torch.testing.assert_close(chunk,full[:,-4:],atol=1e-4,rtol=1e-5)
            assert all(k.shape[2]==x.shape[1] for k,v in updated)
    result=screen({"note":"8. A person reports a cough.\n12. The person is 38 years old.","criteria":["inclusion | Adult."],"mode":"compare"},target,draft)
    assert result["results"][0]["agreement"]
    assert all(e["id"] in {8,12} for e in result["results"][0]["evidence"])
    for payload in [{},{"note":"Test.","criteria":["inclusion |"]},{"note":"Test.","criteria":["rule"]}]:
        try: screen(payload,target,draft)
        except ValueError: continue
        raise AssertionError("Invalid request accepted")
    print("Passed cache offsets, chunked logits, speculative all-match/stop/budget, HTTP payload validation and three-mode equivalence.")
if __name__=="__main__": verify()

