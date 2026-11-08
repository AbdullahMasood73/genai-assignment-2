"""Local browser application: criterion-level screening, evidence, export and live inference lab."""
import json, re, time, webbrowser, sys, hashlib, threading, traceback, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from model import ROOT, LABELS, END, encode, choices, greedy, deployment, load_adapted, MAX_PROMPT, TOKENIZER
from inference import speculative
from prepare import sentences
import benchmark

PORT = 8764
VERSION = "2.0"
MODES = ("cached","uncached","compare","speculative")
LOCK = threading.Lock()          # one inference at a time; HTTP stays responsive

# Explicit, documented aggregation rules. The dataset only labels individual criteria,
# so the overall outcome is a rule-based preliminary summary, never a ground-truth label.
RULES = [
    "R1 Conflict: any inclusion criterion predicted 'not included' or exclusion criterion predicted 'excluded' -> 'Preliminary: potential ineligibility'.",
    "R2 Missing information: otherwise, any 'not enough information' -> 'Preliminary: undetermined - information missing'.",
    "R3 Otherwise -> 'Preliminary: no conflict found among the entered criteria'.",
    "R4 Every decision without a cited sentence is flagged 'unsupported' and listed for manual review; absent evidence is never treated as proof of eligibility or ineligibility.",
    "R5 A 'not enough information' decision that nevertheless cites a sentence is flagged for a consistency check, because the cited text may resolve the criterion.",
]
MEANING = {
    "included": "the patient appears to meet this inclusion requirement",
    "not included": "the patient appears NOT to meet this inclusion requirement",
    "excluded": "the patient appears to have this disqualifying condition",
    "not excluded": "the patient appears not to have this disqualifying condition",
    "not enough information": "the record does not contain enough information to decide this criterion",
    "not applicable": "this criterion does not apply to this patient",
}


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def parse_note(note):
    """Keep original numbered sentence IDs, or number free text from 0."""
    lines=[l for l in note.splitlines() if l.strip()]
    if lines and all(re.match(r"^\s*\d+\.\s+\S",l) for l in lines):
        text=sentences("\n".join(l.strip() for l in lines))
        if len(text)!=len(lines): raise ValueError("Numbered sentences must have unique IDs (for example '0. ...', '1. ...').")
        return text,"\n".join(f"{i}. {s}" for i,s in text.items()),"original numbering"
    parts=[p.strip() for p in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z0-9])",note) if p.strip()]
    text=dict(enumerate(parts))
    return text,"\n".join(f"{j}. {s}" for j,s in text.items()),"numbered automatically from 0"


def parse_criteria(criteria):
    if isinstance(criteria,str): criteria=criteria.splitlines()
    if not isinstance(criteria,list): raise ValueError("Criteria must be a list of lines.")
    criteria=[c for c in criteria if isinstance(c,str) and c.strip()]
    if not 1<=len(criteria)<=30: raise ValueError("Enter between one and thirty criteria, one per line.")
    out=[]
    for n,line in enumerate(criteria,1):
        kind,sep,rule=line.partition("|")
        kind=kind.strip().lower(); rule=rule.strip()
        if kind not in ("inclusion","exclusion") or not sep or not rule:
            raise ValueError(f"Criterion line {n} ('{line.strip()[:60]}') must look like 'inclusion | rule text' or 'exclusion | rule text'.")
        if len(rule)>1000: raise ValueError(f"Criterion line {n} is longer than 1,000 characters.")
        out.append((kind,rule))
    return out


def explain(kind,label,evidence):
    text=f"Predicted '{label}': {MEANING[label]}."
    if evidence:
        many=len(evidence)>1
        text+=(" Cited record sentence"+("s " if many else " ")+", ".join(f"[{e['id']}]" for e in evidence)
               +(" are" if many else " is")+" quoted verbatim below; verify that "+("they" if many else "it")+" actually establish"+("" if many else "es")+" the decision.")
    elif label=="not enough information":
        text+=" No sentence was cited; the missing information should be obtained before deciding."
    else:
        text+=" UNSUPPORTED: no record sentence was cited, so this decision is not grounded in the note and needs manual review."
    return text


def aggregate(rows):
    conflict=[r for r in rows if r["label"] in ("not included","excluded")]
    missing=[r for r in rows if r["label"]=="not enough information"]
    if conflict: overall,rule="Preliminary: potential ineligibility","R1"
    elif missing: overall,rule="Preliminary: undetermined - information missing","R2"
    else: overall,rule="Preliminary: no conflict found among the entered criteria","R3"
    return dict(overall=overall,rule=rule,
                conflicts=[r["criterion"] for r in conflict],
                missing_information=[r["criterion"] for r in missing],
                unsupported=[r["criterion"] for r in rows if r["unsupported"]],
                consistency_checks=[r["criterion"] for r in rows if r.get("check")],
                counts={l:sum(r["label"]==l for r in rows) for l in LABELS})


def screen(payload,target,draft,dep=None,provenance=None):
    if not isinstance(payload,dict): raise ValueError("Request body must be a JSON object.")
    dep=dep or {"end_bias":0.0,"width":4}
    note=payload.get("note","")
    if not isinstance(note,str) or not note.strip(): raise ValueError("Enter a patient note.")
    if len(note)>20000: raise ValueError("The patient note is longer than 20,000 characters.")
    criteria=parse_criteria(payload.get("criteria",[]))
    text,numbered,numbering=parse_note(note.strip())
    if not text: raise ValueError("The note contains no sentences.")
    if max(text)>99: raise ValueError("Use at most 100 sentences with IDs between 0 and 99.")
    mode=payload.get("mode","cached")
    if mode not in MODES: raise ValueError("Unknown decoding mode.")
    rows=[]
    for kind,criterion in criteria:
        ids=encode({"input":{"note":numbered,"type":kind,"criterion":criterion}})
        permitted=lambda out,kind=kind: choices(kind,sorted(text),out)
        times,outputs,draft_stats={}, [], None
        for method in (["uncached","cached","speculative"] if mode=="compare" else [mode]):
            begin=time.perf_counter()
            if method=="speculative":
                seq,accepted,proposed=speculative(target,draft,ids,permitted,width=dep["width"],end_bias=dep["end_bias"])
                draft_stats={"accepted":accepted,"proposed":proposed}
            else:
                seq=greedy(target,ids,permitted,method=="cached",end_bias=dep["end_bias"])
            times[method]=round(1000*(time.perf_counter()-begin),2)
            outputs.append(seq)
        seq=outputs[0]
        evidence=[{"id":s-7,"text":text[s-7]} for s in seq[1:] if s!=END]
        label=LABELS[seq[0]]
        check=("Consistency check: a sentence was cited although the decision is 'not enough information'. "
               "The cited text may in fact resolve this criterion; review it." if evidence and label=="not enough information" else None)
        rows.append(dict(type=kind,criterion=criterion,label=label,evidence=evidence,check=check,
                         unsupported=not evidence and label!="not enough information",
                         explanation=explain(kind,label,evidence),prompt_tokens=len(ids),
                         latency_ms=times,agreement=all(x==seq for x in outputs),draft=draft_stats))
    summary=aggregate(rows)
    return dict(**summary,rules=RULES,results=rows,sentences=[{"id":i,"text":s} for i,s in text.items()],numbering=numbering,
                mode=mode,provenance=dict(provenance or {},generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")),
                notice="Educational screening prototype only. It is not a clinical enrollment decision; every result requires review by a qualified professional.")


def live_benchmark(payload,target,draft,drafts,dep):
    """Run fresh controlled measurements for the Inference Lab view."""
    if not isinstance(payload,dict): raise ValueError("Request body must be a JSON object.")
    try:
        length=int(payload.get("prompt_tokens",256)); budget=int(payload.get("output_tokens",16)); repeats=int(payload.get("repeats",3))
    except (TypeError,ValueError): raise ValueError("Prompt length, output length and repeats must be whole numbers.")
    if not 8<=length<=1024: raise ValueError("Prompt length must be between 8 and 1024 tokens.")
    if not 1<=budget<=64: raise ValueError("Output length must be between 1 and 64 tokens.")
    if not 1<=repeats<=5: raise ValueError("Repeats must be between 1 and 5.")
    name=payload.get("draft",dep["draft"])
    if name not in drafts: raise ValueError("Unknown draft model.")
    source=payload.get("text","")
    if isinstance(source,str) and source.strip():
        prompt=TOKENIZER.encode(source).ids
        while len(prompt)<length: prompt=prompt+prompt
        prompt=prompt[:length]; origin="text you entered (repeated to reach the length)"
    else:
        prompt=benchmark.long_prompt(length); origin="validation-split prompts"
    rows,margin=benchmark.decode_paths(target,drafts[name],prompt,budget,repeats=repeats,width=dep["width"])
    return dict(rows=rows,min_top2_margin=margin,prompt_source=origin,draft=name,repeats=repeats,
                protocol="1 warm-up + median of repeats; stop symbol suppressed so all paths emit exactly the requested tokens; equality checked against the uncached reference.")


def live_attention(payload):
    if not isinstance(payload,dict): raise ValueError("Request body must be a JSON object.")
    try:
        lq=int(payload.get("q_len",128)); lk=int(payload.get("k_len",128)); block=int(payload.get("block",32))
    except (TypeError,ValueError): raise ValueError("Attention sizes must be whole numbers.")
    if not (1<=lq<=1024 and 1<=lk<=1024 and lq<=lk): raise ValueError("Use 1 <= query length <= key length <= 1024.")
    if not 4<=block<=256: raise ValueError("Block size must be between 4 and 256.")
    return dict(**benchmark.attention_check(lq,lk,bool(payload.get("causal",True)),block,repeats=3),stability=benchmark.stability_check())


class Server(ThreadingHTTPServer):
    # SO_REUSEADDR on Windows lets a second server silently share the port; bind exclusively.
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        import socket
        if hasattr(socket,"SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n"%(self.log_date_time_string(),fmt%args))

    def reply(self,status,body,kind="application/json"):
        data=body if isinstance(body,bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type",kind)
        self.send_header("Cache-Control","no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        s=self.server
        if self.path in ("/","/index.html"):
            return self.reply(200,(ROOT/"index.html").read_bytes(),"text/html; charset=utf-8")
        if self.path=="/health":
            return self.reply(200,dict(status="ready",version=VERSION,**s.provenance))
        if self.path=="/logs":
            # Offline experiment logs, shown alongside (not instead of) live measurements.
            logs={}
            for name in ["benchmark_log","test_metrics_summary","history"]:
                p=ROOT/"results"/(name+".json")
                if p.exists(): logs[name]=json.loads(p.read_text())
            return self.reply(200,logs)
        self.send_error(404)

    def do_POST(self):
        s=self.server
        try:
            size=int(self.headers.get("Content-Length") or 0)
            if size>200000: raise ValueError("Request is too large (limit 200 kB).")
            try: payload=json.loads(self.rfile.read(size) or b"{}")
            except json.JSONDecodeError: raise ValueError("Request body is not valid JSON.")
            with LOCK:
                if self.path=="/screen": out=screen(payload,s.target,s.draft,s.dep,s.provenance)
                elif self.path=="/benchmark": out=live_benchmark(payload,s.target,s.draft,s.drafts,s.dep)
                elif self.path=="/attention": out=live_attention(payload)
                else: return self.send_error(404)
            status=200
        except (ValueError,TypeError,AttributeError,KeyError) as e:
            out,status={"error":str(e)},400
        except Exception as e:      # never leave the browser without an answer
            traceback.print_exc()
            out,status={"error":f"Internal error ({type(e).__name__}). The server is still running; check the input and retry."},500
        self.reply(status,out)


def main():
    url=f"http://127.0.0.1:{PORT}/"
    try:
        server=Server(("127.0.0.1",PORT),Handler)
    except OSError:
        from urllib.request import urlopen
        try:
            with urlopen(url+"health",timeout=3) as response: ok=b"ready" in response.read()
        except Exception: ok=False
        if not ok:
            print(f"Port {PORT} is used by another program. Close it and start again."); sys.exit(1)
        print("The workbench is already running.");
        if "--headless" not in sys.argv: webbrowser.open(url)
        return
    print("Loading models...",flush=True)
    dep=deployment()
    server.dep=dep
    server.target=load_adapted(dep["target"])
    server.drafts={name:load_adapted(spec) for name,spec in dep["drafts"].items()}
    server.draft=server.drafts[dep["draft"]]
    server.provenance=dict(app_version=VERSION,target=dep["target_name"],target_checkpoint_sha256=sha(dep["target"]["checkpoint"])[:16],
                           draft=dep["draft"],end_bias=dep["end_bias"],draft_width=dep["width"],
                           base_revision=(ROOT/dep["target"]["base"]/"revision.txt").read_text().strip()[:12],
                           max_prompt_tokens=MAX_PROMPT,decoding="constrained greedy")
    print(f"Ready: {url}  (keep this window open; close it to stop the server)",flush=True)
    if "--headless" not in sys.argv: webbrowser.open(url)
    server.serve_forever()


if __name__=="__main__": main()
