import json
from pathlib import Path
p=Path(__file__).resolve().parent
def get(name):return json.loads((p.parent/'results'/(name+'.json')).read_text())
out=[]
def table(label,title,header,rows,comment):
    out.extend([r'\begin{table}[ht]\centering\caption{'+title+r'}\label{tab:'+label+r'}\small\begin{tabular}{'+'l'+'r'*(len(header)-1)+'}',r'\hline',' & '.join(header)+r'\\\hline'])
    out.extend(' & '.join(map(str,row))+r'\\' for row in rows)
    out.extend([r'\hline\end{tabular}\end{table}',r'Table~\ref{tab:'+label+'} '+comment])
rows=[]
for split in ['validation','test']:
    for model in ['base','adapted']:
        v=get(split+'_'+model+'_metrics')['all']
        rows.append([split.title(),model,*[f'{x:.3f}' for x in (v['accuracy'],v['classification']['macro avg']['f1-score'],v['evidence_micro'][2],v['evidence_macro'][2])]])
table('scores','Paired classification and evidence scores',['Split','Model','Acc.','Macro-F1','Ev. micro','Ev. macro'],rows,'shows that supervised adaptation improves classification on both splits while eliminating positive evidence retrieval. The high macro evidence score arises from correctly empty sets.')
v=get('test_adapted_metrics')
rows=[]
for group,d in v.items():
    rows.append([group,d['n'],f"{d['accuracy']:.3f}",f"{d['classification']['macro avg']['f1-score']:.3f}"])
table('type','Adapted test classification by criterion type',['Group','Rows','Accuracy','Macro-F1'],rows,'shows stronger exclusion accuracy. The model frequently predicts not excluded in this group.')
d=v['all']; rows=[]
for label in d['labels']:
    s=d['classification'][label]
    rows.append([label,int(s['support']),*['%.3f'%s[k] for k in ['precision','recall','f1-score']]])
table('classes','Adapted test performance by expert category',['Class','Support','Precision','Recall','F1'],rows,'shows that all correctly classified cases come from two labels. Excluded has no test support. Full per-type tables and confusion matrices are included as machine-readable metrics.')
table('matrix','Adapted test confusion matrix; class order follows Table~\\ref{tab:classes}',['Gold / pred.','A','B','C','D','E','F'],[[chr(65+i),*row] for i,row in enumerate(d['confusion'])],'locates the largest systematic mistakes. Every included and not-included example is assigned insufficient information.')
rows=[]
for r in get('decoding'):
    rows.append([f"{r['input']}/{r['output']}",r['mode'],f"{1000*r['prefill_s']:.1f}" if 'prefill_s' in r else '--',f"{1000*r['decode_s']:.1f}" if 'decode_s' in r else '--',f"{r['tokens_s']:.1f}",f"{r['peak_rss_mib']:.0f}"])
table('runtime','Generation timing in milliseconds and peak process RSS',['In/out','Path','Prefill','Decode','Tok/s','MiB'],rows,'shows token agreement in all cases (logs include the assertions). Caching helps most on the longest prefix. Draft acceptance was 1/7, 2/22 and 7/39. Verification overhead made speculation slower than cached greedy at all three lengths.')
rows=[]
for r in get('attention'):
    rows.append([r['length'],'yes' if r['causal'] else 'no',f"{r['max_error']:.1e}",f"{r['mean_error']:.1e}",f"{1000*r['reference_s']:.2f}",f"{1000*r['tiled_s']:.2f}"])
table('attention','Attention agreement and CPU runtime (milliseconds)',['Length','Causal','Max err.','Mean err.','Full','Tiled'],rows,'shows FP32 agreement across causal and unmasked shapes. Python block dispatch dominates at these lengths. The score tile has at most 32 by 32 entries per head; process RSS includes retained buffers and does not directly measure this saving.')
(p/'tables.tex').write_text('\n'.join(out))
