"""Generate numbers.tex (macros), tables.tex and figures.tex from results/*.json.

Every number quoted in the paper comes from these files, so text and logs cannot drift.
Run from the project root or the report folder: python report/tables.py
"""
import json, statistics
from pathlib import Path

P=Path(__file__).resolve().parent
R=P.parent/"results"
get=lambda n: json.loads((R/(n+".json")).read_text())
f3=lambda x: "--" if x is None else f"{x:.3f}"
f1=lambda x: "--" if x is None else f"{x:.1f}"
SHORT={"included":"Incl.","not included":"Not incl.","excluded":"Excl.","not excluded":"Not excl.","not enough information":"NEI","not applicable":"N/A"}
NAMES={"majority":"Majority","lead":"Maj.+lead sent.","first":"1st subm. 14M","base":"Base 70M","adapted0":"Adapted, $b{=}0$","adapted":"Adapted (final)"}

macros={}
def macro(name,value): macros[name]=value

tables=[]
def sci(x):
    m,e=f"{x:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"
def table(label,caption,cols,header,rows,wide=False,note=None):
    env="table*" if wide else "table"
    t=[rf"\begin{{{env}}}[t]\centering\caption{{{caption}}}\label{{tab:{label}}}\footnotesize",
       rf"\begin{{tabular}}{{{cols}}}",r"\hline"," & ".join(header)+r"\\\hline"]
    t+=[" & ".join(map(str,r))+r"\\" for r in rows]
    t+=[r"\hline\end{tabular}"]
    if note: t+=[rf"\\[2pt]\parbox{{{'0.95' if wide else '0.98'}\linewidth}}{{\scriptsize {note}}}"]
    t+=[rf"\end{{{env}}}"]
    tables.append("\n".join(t))

# ---------------- training ----------------
cfg=get("train_config"); hist=get("history"); sel=get("selected")
table("hparams","Fine-tuning configuration (Pythia-70M target)","ll",["Setting","Value"],[
    ["Trainable / total parameters",f"{cfg['trainable_parameters']:,} / {cfg['total_parameters']:,}"],
    ["Frozen","input embedding (25.8M)"],
    ["Objective","answer-only teacher-forced CE"],
    ["Label weights",f"balanced, power {cfg['balance_power']:g}"],
    ["Evidence-ID weight",f"{cfg['evidence_weight']:g} (stop symbol 1)"],
    ["Optimiser","AdamW, wd 0.01, clip 1.0"],
    ["Learning rate",f"{cfg['lr']:g}, 6\\% warm-up, linear decay"],
    ["Batch","1 record, accumulate {}".format(cfg["accumulate"])],
    ["Epochs / selected",f"{cfg['epochs']} / {sel['epoch']}"],
    ["Precision / threads / seed",f"FP32 / {cfg['threads']} CPU / {cfg['seed']}"],
])
macro("SelEpoch",sel["epoch"]); macro("Epochs",cfg["epochs"])
macro("EpochMinutes",f"{statistics.mean(h['train_seconds'] for h in hist[1:])/60:.1f}")
macro("TrainParams",f"{cfg['trainable_parameters']/1e6:.2f}M")

figs=[]
pts=lambda key,rows: " ".join(f"({h['epoch']},{h[key]:.4f})" for h in rows)
figs.append(r"""\begin{figure}[t]\centering
\begin{tikzpicture}\begin{axis}[width=\columnwidth,height=4.6cm,xlabel={Epoch},ylabel={Validation score},ymin=0,ymax=0.75,
legend style={font=\scriptsize,at={(0.5,1.03)},anchor=south,legend columns=3,draw=none},tick label style={font=\scriptsize},label style={font=\scriptsize},
axis y line*=left,xmin=0,xmax=%d]
\addplot[thick,mark=*,color=teal] coordinates {%s};\addlegendentry{macro-F1 (present)}
\addplot[thick,mark=square*,color=orange] coordinates {%s};\addlegendentry{evidence micro-F1}
\addplot[thick,mark=triangle*,color=gray] coordinates {%s};\addlegendentry{accuracy}
\end{axis}
\begin{axis}[width=\columnwidth,height=4.6cm,axis y line*=right,axis x line=none,ylabel={Train loss},ymin=0,xmin=0,xmax=%d,
tick label style={font=\scriptsize},label style={font=\scriptsize}]
\addplot[dashed,thick,color=black] coordinates {%s};
\end{axis}\end{tikzpicture}
\caption{Validation metrics per epoch (solid, left axis; epoch 0 is the unadapted base) and mean training loss (dashed, right axis). The checkpoint with the best selection score (epoch %d) was frozen.}\label{fig:train}\end{figure}"""%(
    cfg["epochs"],pts("macro_f1_present",hist),pts("evidence_micro_f1",hist),pts("accuracy",hist),cfg["epochs"],pts("loss",hist[1:]),sel["epoch"]))

# ---------------- calibration ----------------
cal=get("calibration")
macro("EndBias",f"{cal['selected']:g}")
figs.append(r"""\begin{figure}[t]\centering
\begin{tikzpicture}\begin{axis}[width=\columnwidth,height=4cm,xlabel={Stop-symbol bias $b$},ylabel={Validation evidence},ymin=0,ymax=1,
legend style={font=\scriptsize,at={(0.02,0.97)},anchor=north west},tick label style={font=\scriptsize},label style={font=\scriptsize}]
\addplot[thick,mark=*,color=teal] coordinates {%s};\addlegendentry{micro-F1}
\addplot[thick,mark=square*,color=orange] coordinates {%s};\addlegendentry{micro precision}
\addplot[thick,mark=triangle*,color=gray] coordinates {%s};\addlegendentry{micro recall}
\end{axis}\end{tikzpicture}
\caption{Validation-only sweep of the additive stop-symbol bias. Lower $b$ makes the decoder cite more sentences, trading precision for recall; $b=%g$ maximises micro-F1 and was frozen before testing.}\label{fig:cal}\end{figure}"""%(
    " ".join(f"({s['end_bias']},{s['evidence_micro'][2]:.4f})" for s in cal["sweep"]),
    " ".join(f"({s['end_bias']},{s['evidence_micro'][0]:.4f})" for s in cal["sweep"]),
    " ".join(f"({s['end_bias']},{s['evidence_micro'][1]:.4f})" for s in cal["sweep"]),cal["selected"]))

# ---------------- main scores ----------------
summ=get("metrics_summary")
import sys; sys.path.insert(0,str(P.parent))
from evaluate import report as _report, summary_row as _row
from train import read as _read
for split in ["validation","test"]:
    pr=json.loads((R/"first_submission"/f"{split}_adapted_predictions.json").read_text())
    summ.append(_row(split,"first",_report(_read(split),pr)))
order={"majority":0,"lead":1,"first":2,"base":3,"adapted0":4,"adapted":5}
summ=sorted(summ,key=lambda s:(s["split"]!="validation",order[s["model"]]))
if cal["selected"]==0: summ=[s for s in summ if s["model"]!="adapted0"]
rows=[]
for s in summ:
    rows.append([s["split"].title(),NAMES[s["model"]],f3(s["accuracy"]),f3(s["macro_f1"]),f3(s["macro_f1_present"]),
                 f3(s["inclusion_accuracy"]),f3(s["exclusion_accuracy"]),f3(s["evidence_micro_p"]),f3(s["evidence_micro_r"]),f3(s["evidence_micro_f1"]),
                 f3(s["evidence_macro_f1"]),f3(s["evidence_macro_f1_nonempty"])])
table("scores","Paired before/after comparison on identical validation and held-out test rows (constrained greedy decoding)","llcccccccccc",
      ["Split","System","Acc.","MF1$_6$","MF1$_p$","Acc$_{inc}$","Acc$_{exc}$","Ev.P","Ev.R","Ev.F1$_\\mu$","Ev.F1$_M$","Ev.F1$_M^{+}$"],rows,wide=True,
      note=r"MF1$_6$: macro-F1 over all six labels (zero-support classes count as 0); MF1$_p$: over labels present in the gold set. Ev.\ = evidence sentence retrieval; $\mu$ micro (pooled), $M$ per-record macro where two empty sets score 1, $M^{+}$ macro over records with non-empty gold evidence only. Majority = type-conditioned training majority label without evidence; Maj.+lead sent.\ adds the first note sentence as evidence; 1st subm.\ = our original Pythia-14M submission re-scored with these metrics.")
T={(s["split"],s["model"]):s for s in summ}
for (split,model),s in T.items():
    tag=split.title()+{"majority":"Maj","lead":"Lead","first":"First","base":"Base","adapted0":"Raw","adapted":"Ft"}[model]
    for k,v in [("Acc","accuracy"),("MF","macro_f1"),("MFp","macro_f1_present"),("EvF","evidence_micro_f1"),("EvP","evidence_micro_p"),("EvR","evidence_micro_r"),("EvM","evidence_macro_f1"),("EvMp","evidence_macro_f1_nonempty")]:
        macro(tag+k,f3(s[v]))
    macro(tag+"AccPct",f"{100*s['accuracy']:.1f}\\%")

# ---------------- per class + confusion ----------------
m=get("test_adapted_metrics")
for kind in ["inclusion","exclusion"]:
    d=m[kind]; L=d["labels"]
    rows=[]
    for i,l in enumerate(L):
        c=d["classification"][l]
        rows.append([SHORT[l],int(c["support"]),f3(c["precision"]),f3(c["recall"]),f3(c["f1-score"])]+d["confusion"][i])
    table("cls_"+kind,f"Adapted model, held-out test, {kind} criteria ($n={d['n']}$): per-class scores and confusion matrix (rows gold, columns predicted)",
          "lrccc|"+"r"*len(L),["Gold","n","P","R","F1"]+[SHORT[l] for l in L],rows)
    macro(kind.title()+"N",d["n"]); macro(kind.title()+"Acc",f3(d["accuracy"])); macro(kind.title()+"MF",f3(d["macro_f1"])); macro(kind.title()+"MFp",f3(d["macro_f1_present"]))
d=m["all"]
macro("TestN",d["n"]); macro("TestCorrect",int(round(d["accuracy"]*d["n"])))
macro("GoldEvRecords",d["records_with_gold_evidence"]); macro("PredEvRecords",d["records_with_predicted_evidence"])
b=get("test_base_metrics")["all"]
rows=[]
for l in d["labels"]:
    rows.append([SHORT[l],int(d["classification"][l]["support"]),f3(b["classification"][l]["f1-score"]),f3(d["classification"][l]["f1-score"]),
                 b["predicted_distribution"].get(l,0),d["predicted_distribution"].get(l,0)])
table("classes","Held-out test per-label F1 and prediction counts before (base) and after adaptation","lrcccc",["Label","Support","F1 base","F1 adapt.","\\#pred base","\\#pred adapt."],rows)

# ---------------- uncertainty (patient-level bootstrap) ----------------
un=get("test_uncertainty")
ci=lambda d: f"{d['point']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
rows=[]
for k,lab in [("adapted-majority","vs.\\ majority"),("adapted-lead","vs.\\ maj.+lead"),("adapted-base","vs.\\ base 70M")]:
    d=un["differences"][k]; rows.append([lab]+[ci(d[m]) for m in ["accuracy","macro_f1_present","evidence_micro_f1"]])
mc=un["mcnemar_adapted_vs_majority"]
table("ci",f"Test difference of the adapted model with patient-level 95\\% bootstrap intervals ({un['draws']} resamples of the 8 test patients)","lccc",
      ["Adapted","$\\Delta$Acc.","$\\Delta$MF1$_p$","$\\Delta$Ev.F1$_\\mu$"],rows,wide=True,
      note=f"Exact McNemar test on label correctness, adapted vs.\\ majority: {mc['adapted_only_correct']} vs.\\ {mc['majority_only_correct']} discordant criteria, $p={mc['exact_p']:.2f}$.")
macro("McnemarP",f"{mc['exact_p']:.2f}"); macro("McnemarA",mc["adapted_only_correct"]); macro("McnemarM",mc["majority_only_correct"])

# ---------------- decoding ----------------
bl=get("benchmark_log"); dep=bl["deployment"]
rows=[]
for r in bl["fixed_length"]:
    mode=r["mode"]+(" ("+("early-exit" if r["draft"]=="early_exit" else "14M")+")" if r["mode"]=="speculative" else "")
    rows.append([f"{r['input']}/{r['output']}",mode,f1(r["prefill_ms"]),f1(r["decode_ms"]),f1(r["total_ms"]),f1(r["tokens_s"]),
                 f"{r['speedup_vs_uncached']:.2f}",f"{r['peak_rss_mib']:.0f}",f"{r['kv_cache_kib']:.0f}" if r.get("kv_cache_kib") else "--",
                 f"{r['accepted']}/{r['proposed']}" if r["mode"]=="speculative" else "--","yes" if r["equal"] else "NO"])
table("runtime","Controlled greedy decoding on the adapted 70M target (stop suppressed; warm-up + median of 5; CPU, %d threads)"%bl["environment"]["threads"],
      "llrrrrrrrrc",["In/out","Path","Prefill","Decode","Total ms","Tok/s","$\\times$unc.","RSS MiB","KV KiB","Acc./prop.","Equal"],rows,wide=True,
      note="Prefill and decode are timed separately for the two non-speculative paths; speculative time is end-to-end (both prefills included). KV KiB is the analytic FP32 cache size at the end of generation. RSS is sampled process memory (interpreter, all models) and can miss short peaks. Equal: token sequence identical to the uncached reference.")
fx=bl["fixed_length"]
def pick(inp,mode,draft=None):
    return next(r for r in fx if r["input"]==inp and r["mode"]==mode and (draft is None or r["draft"]==draft))
for inp,tag in [(32,"S"),(256,"M"),(1024,"L")]:
    macro("Cache"+tag,f"{pick(inp,'cached')['speedup_vs_uncached']:.2f}")
    for dn,dt in [("early_exit","EE"),("pythia14m","Sm")]:
        if any(r["input"]==inp and r["mode"]=="speculative" and r["draft"]==dn for r in fx):
            r=pick(inp,"speculative",dn); macro("Spec"+dt+tag,f"{r['speedup_vs_cached']:.2f}"); macro("Acc"+dt+tag,f"{100*r['acceptance']:.0f}\\%")
macro("MinMargin",f"{min(r['min_top2_margin'] for r in fx if r.get('min_top2_margin') is not None):.3f}")

# screening workload
sc=bl["screening"]
rows=[]
for mode,draft in [("uncached",None),("cached",None),("speculative","early_exit"),("speculative","pythia14m")]:
    rs=[r for r in sc if r["mode"]==mode and (draft is None or r["draft"]==draft)]
    if not rs: continue
    acc=sum(r.get("accepted") or 0 for r in rs); prop=sum(r.get("proposed") or 0 for r in rs)
    rows.append([mode+(" ("+("early-exit" if draft=="early_exit" else "14M")+")" if draft else ""),len(rs),
                 f1(statistics.mean(r["total_ms"] for r in rs)),f"{statistics.mean(r['output'] for r in rs):.2f}",
                 f"{acc}/{prop}" if prop else "--",f"{100*acc/prop:.0f}\\%" if prop else "--",f"{sum(r['equal'] for r in rs)}/{len(rs)}"])
table("screening","Real screening outputs (grammar and stop enabled) on 24 validation criteria: mean latency per criterion","lrrrrrr",
      ["Path","n","Mean ms","Out tok.","Accepted","Rate","Equal"],rows)

# ---------------- attention ----------------
rows=[]
for r in bl["attention"]:
    rows.append([f"{r['q_len']}$\\times${r['k_len']}","y" if r["causal"] else "n",r["block"],f"{r['ref_max_error']:.1e}",f"{r['tiled_max_error']:.1e}",f"{r['tiled_mean_error']:.1e}",
                 f"{r['reference_ms']:.2f}",f"{r['tiled_ms']:.2f}",f"{r['sdpa_ms']:.2f}",f"{r['full_scores_kib']:.0f}",f"{r['tile_scores_kib']:.0f}"])
table("attention","Attention kernels vs.\\ an FP64 SDPA ground truth (4 heads, $d_h=32$, FP32, median of 5)","lccrrrrrrrr",
      ["$L_q\\times L_k$","C","Blk","Ref max","Tiled max","Tiled mean","Ref ms","Tiled ms","SDPA ms","Full KiB","Tile KiB"],rows,wide=True,
      note="Errors are maximum/mean absolute differences from FP64 \\texttt{scaled\\_dot\\_product\\_attention}. Full/Tile KiB: size of the score matrix materialised by the reference versus the largest single tile of the online-softmax kernel. Rectangular shapes ($L_q<L_k$) model decoding with a KV cache.")
att=bl["attention"]
macro("AttnMaxErr",sci(max(r['tiled_max_error'] for r in att)))
macro("AttnRefMaxErr",sci(max(r['ref_max_error'] for r in att)))
macro("TiledSlow",f"{statistics.median(r['tiled_ms']/r['reference_ms'] for r in att):.1f}")
macro("SdpaFast",f"{statistics.median(r['reference_ms']/r['sdpa_ms'] for r in att):.1f}")

# ---------------- reference audit ----------------
aud=get("reference_audit")
macro("AuditMax",sci(max(a['max_abs_logit_error'] for a in aud)))
macro("AuditPositions",sum(a["same_argmax_positions"] for a in aud)); macro("AuditTotal",sum(a["length"] for a in aud))

# ---------------- distillation ----------------
dl=get("distill_log")
macro("DistillBefore",f"{100*dl[0]['validation_agreement']:.0f}\\%"); macro("DistillAfter",f"{100*dl[-1]['validation_agreement']:.0f}\\%")

# ---------------- diagnostics ----------------
dg=get("diagnostics")
for k,v in dg.items():
    macro("Diag"+k.replace("_",""),f"{v['mean_label_auc']:.2f}"); macro("DiagEv"+k.replace("_",""),f"{v['evidence_auc']:.2f}")

(P/"numbers.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k,v in macros.items())+"\n")
(P/"tables.tex").write_text("\n\n".join(tables)+"\n")
(P/"figures.tex").write_text("\n\n".join(figs)+"\n")
print(len(macros),"macros,",len(tables),"tables,",len(figs),"figures")

# ---------------- experiment history and diagnostics (appended) ----------------
ex=get("experiments_summary")
def short(t):
    t=t.replace(" (original submission)"," (1st subm.)").replace("balanced^","bal.\\,$p{=}$").replace("unweighted","unw.")
    t=t.replace(", lr ",", ").replace(" ep","\\,ep").replace(" x"," $\\times$").replace("ev+stop","ev+stop")
    return t
rows=[[short(d["description"]),
       d["epochs_completed"],d["best_epoch"],f3(d["accuracy"]),f3(d["macro_f1_present"]),f3(d["evidence_micro_f1"]),f3(d["selection"])] for d in ex.values()]
table("runs","All adaptation runs, validation split only (best epoch by the selection score)","lrrrrrr",["Configuration","Ep.","Best","Acc.","MF1$_p$","Ev.F1","Sel."],rows,
      note="``ev+stop'' runs used an early weighting bug that also up-weighted the stop symbol; 14M runs were stopped early once collapse was evident. The last row is the frozen model.")
rows=[]
for k,lab in [("SmallBase","Pythia-14M base"),("SmallFirst","Pythia-14M 1st subm."),("LargeBase","Pythia-70M base"),("LargeFinal","Pythia-70M final")]:
    d=dg[k]; rows.append([lab,f"{d['mean_label_auc']:.2f}",f"{d.get('inclusion:included',0):.2f}",f"{d.get('exclusion:not applicable',0):.2f}",f"{d['evidence_auc']:.2f}"])
table("diag","Latent signal on validation from teacher-forced logits (ROC-AUC; 0.5 = chance)","lrrrr",["Model","Mean label","Included","Excl. N/A","Evidence"],rows)
(P/"numbers.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k,v in macros.items())+"\n")
tables=tables[-2:]+tables[:-2]
(P/"tables.tex").write_text("\n\n".join(tables)+"\n")
print(len(tables),"tables after appendix")
