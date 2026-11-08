import json,re,time,webbrowser,sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from model import ROOT,LABELS,END,Model,encode,choices,greedy
from benchmark import loaded
from inference import speculative
from prepare import sentences

def screen(payload,target,draft):
    note=payload.get("note","").strip()
    criteria=payload.get("criteria",[])
    if not note or not isinstance(criteria,list) or not 1<=len(criteria)<=30:
        raise ValueError("Enter a patient note and between one and thirty criteria.")
    lines=note.splitlines()
    if all(re.match(r"^\d+\.\s+",line) for line in lines):
        text=sentences(note)
        if len(text)!=len(lines): raise ValueError("Numbered sentences must have unique IDs.")
    else:
        parts=[p.strip() for p in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])",note) if p.strip()]
        text=dict(enumerate(parts))
        note="\n".join(f"{j}. {s}" for j,s in text.items())
    if not text or max(text)>99: raise ValueError("Use sentence IDs between 0 and 99.")
    mode=payload.get("mode","cached")
    if mode not in ("cached","uncached","compare","speculative"): raise ValueError("Unknown decoding mode.")
    rows=[]
    for line in criteria:
        kind,sep,criterion=line.partition("|")
        kind=kind.strip().lower()
        criterion=criterion.strip()
        if kind not in ("inclusion","exclusion") or not sep or not criterion: raise ValueError("Each criterion needs inclusion or exclusion, a vertical bar, and rule text.")
        ids=encode({"input":{"note":note,"type":kind,"criterion":criterion}})
        permitted=lambda out: choices(kind,sorted(text),out)
        times={}
        outputs=[]
        draft_stats=None
        for method in (["uncached","cached","speculative"] if mode=="compare" else [mode]):
            begin=time.perf_counter()
            if method=="speculative":
                seq,accepted,proposed=speculative(target,draft,ids,permitted)
                draft_stats={"accepted":accepted,"proposed":proposed}
            else: seq=greedy(target,ids,permitted,method=="cached")
            times[method]=round(1000*(time.perf_counter()-begin),2)
            outputs.append(seq)
        seq=outputs[0]
        evidence=[{"id":s-7,"text":text[s-7]} for s in seq[1:] if s!=END]
        label=LABELS[seq[0]]
        explanation=("The model identified insufficient information for this rule." if label=="not enough information" else f"The model classified this {kind} criterion as {label}.")
        explanation+=(" Review the cited sentences against the full criterion." if evidence else " No supporting sentence was retrieved. Manual review is needed.")
        rows.append(dict(type=kind,criterion=criterion,label=label,evidence=evidence,explanation=explanation,latency_ms=times,agreement=all(x==seq for x in outputs),draft=draft_stats))
    if any(r["label"] in ("not included","excluded") and r["evidence"] for r in rows): overall="Potential conflict — review required"
    elif any(r["label"]=="not enough information" or not r["evidence"] for r in rows): overall="Insufficient evidence — review required"
    else: overall="No flagged conflict among entered criteria — review required"
    return dict(overall=overall,results=rows,model="Pythia-14M / TrialGPT adaptation",notice="Educational screening prototype. Clinical enrollment decisions require professional review.")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path!="/": return self.send_error(404)
        data=(ROOT/"index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type","text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(data)
    def do_POST(self):
        if self.path!="/screen": return self.send_error(404)
        try:
            size=int(self.headers.get("Content-Length",0))
            if size>100000: raise ValueError("Input too long.")
            out=screen(json.loads(self.rfile.read(size)),self.server.target,self.server.draft)
            status=200
        except (ValueError,TypeError,AttributeError) as e:
            out={"error":str(e)}
            status=400
        self.send_response(status)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())

if __name__=="__main__":
    try:
        server=HTTPServer(("127.0.0.1",8764),Handler)
    except OSError:
        from urllib.request import urlopen
        with urlopen("http://127.0.0.1:8764/",timeout=2) as response:
            if b"<title>Trial screening workbench</title>" not in response.read(1000): raise
        if "--headless" not in sys.argv: webbrowser.open("http://127.0.0.1:8764/")
        sys.exit(0)
    server.target,server.draft=loaded(),Model(2).eval()
    if "--headless" not in sys.argv: webbrowser.open("http://127.0.0.1:8764/")
    server.serve_forever()

