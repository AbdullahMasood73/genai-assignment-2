# Trial screening workbench

Muhammad Abdullah Masood · I230756 · Generative AI, Assignment 2

A CPU-only, locally executed Transformer system that matches a patient note to individual clinical-trial inclusion/exclusion criteria, returns a label from the dataset's full label set together with the **original patient-sentence IDs** that support it, and demonstrates hand-written **KV-cached decoding**, **tiled online-softmax attention** and **greedy speculative decoding**, all verified token-for-token. A browser application exposes the screening workflow, report export and live inference benchmarks.

> Educational screening prototype. It is **not** a clinical enrollment decision tool.

## Where to find everything

| Item | Location |
|---|---|
| Assignment question | [GenAI_Assignment#2.pdf](GenAI_Assignment%232.pdf) (local folder / submission ZIP) |
| IEEE report (PDF) | [report/paper.pdf](report/paper.pdf) |
| LaTeX source | [report/](report/) - `paper.tex`, `body.tex`, `results.tex`, `figures_arch.tex`; `numbers.tex`, `tables.tex`, `figures.tex` are generated from `results/` by `report/tables.py` |
| Browser application | [Start-Screening.cmd](Start-Screening.cmd) → `app.py` + `index.html` |
| Demonstration video | [demo.mp4](demo.mp4) |
| Split manifest | [data/manifest.json](data/manifest.json) |
| Trained weights | `results/adaptation.pt` (Pythia-70M target), `results/draft14m.pt` (distilled draft) |
| Logs and metrics | [results/](results/) (see table below) |

## Results (held-out test: 8 patients, 206 criteria)

| System | Accuracy | Macro-F1 (present labels) | Evidence micro-F1 |
|---|---|---|---|
| Majority label per criterion type | 0.626 | 0.272 | 0.000 |
| Majority label + first sentence as evidence | 0.626 | 0.272 | 0.253 |
| First submission (Pythia-14M) | 0.612 | 0.266 | 0.000 |
| Pythia-70M before fine-tuning | 0.194 | 0.173 | 0.119 |
| **Pythia-70M fine-tuned (final)** | **0.631** | **0.334** | **0.275** |

Honest reading: patient-level bootstrap intervals (`results/test_uncertainty.json`) show the fine-tuned model is significantly better than the unadapted Pythia-70M on every metric. On accuracy it is statistically tied with the majority rule (McNemar p = 1.00), and on evidence with the lead-sentence rule. It predicts "not enough information" for every inclusion criterion, and every citation is sentence 0. The test patients had already been used to score the first submission, so treat them as held-out rather than pristine. See the report's Results, Error-analysis and Limitations sections.

Systems results (all paths token-identical to the uncached reference):
- KV cache: 2.9× / 9.3× / 23.7× faster than uncached at 32/256/1,024 prompt tokens.
- Tiled online-softmax attention: max error 6.1e-7 vs FP64; exact but 7.9× slower than the reference in Python.
- Speculative decoding (early-exit and distilled 14M drafts): exact, but not reliably faster than cached decoding on this CPU; the report explains why.

## Start the application (no coding needed)

1. Extract the whole submission ZIP to a writable folder on Windows.
2. Install 64-bit **Python 3.12** (3.13 also works; 3.11 and 3.14 do not, because the pinned numpy/torch wheels do not exist for them) from python.org if it is not installed.
3. Double-click **Start-Screening.cmd**. On the first run it creates `.venv`, installs `requirements.txt`, and downloads (if absent) and SHA-256-verifies the two pinned Pythia base checkpoints. Allow several minutes and internet access.
4. The browser opens http://127.0.0.1:8764 when the models are loaded. Keep the launcher window open; close it to stop the server.

Using it:
- **Screening tab**: paste a note (numbered lines `0. …` keep their IDs; free text is split and numbered from 0) and criteria, one per line, as `inclusion | text` or `exclusion | text`. Pick a decoding mode (*Compare all three modes* runs uncached, cached and speculative decoding and checks that they agree). Results show the label, verbatim evidence, the explanation, flags (unsupported decision, consistency check) and an overall **preliminary** outcome from explicit rules R1-R5. Export as HTML or JSON with provenance.
- **Inference lab tab**: fresh, warmed, repeated benchmarks for any prompt/output length and draft, plus a live attention-kernel check against FP64.
- **Rules & provenance tab**: aggregation rules, label meanings, loaded checkpoint hash and base revision.

The server listens on loopback only and keeps notes in memory. Use fictional records.

## Reproduce the experiments

From the project folder with the environment active (`.venv\Scripts\activate`):

    python setup_model.py                      # download + verify Pythia-14M and Pythia-70M bases
    python prepare.py                          # 37/8/8 patient split, JSONL files, manifest
    python train.py --base base70 --epochs 16 --lr 2e-4 --schedule linear --class-weight balanced --balance-power 1 --evidence-weight 5 --threads 6 --out experiments/p70_e16
    copy experiments\p70_e16\adaptation.pt results\   (and history/selected/train_config/validation_*.json)
    python evaluate.py calibrate               # stop-symbol bias on VALIDATION only
    python distill.py 4 7                      # distil the Pythia-14M draft on the target's training outputs
    python evaluate.py final                   # writes freeze.json, then scores validation and test once
    python evaluate.py baselines               # rule-based lead-sentence baseline
    python evaluate.py uncertainty             # patient-level bootstrap CIs + McNemar (saved predictions only)
    python diagnose.py --all                   # latent label/evidence signal (validation)
    python benchmark.py                        # decoding + attention benchmarks -> results/benchmark_log.json
    python verify.py                           # acceptance tests (6 groups)
    python report/tables.py                    # regenerate the paper's numbers/tables/figures
    .tools\tectonic.exe report\paper.tex       # or any XeLaTeX/LaTeX editor

Optional: `audit_reference.py` compares the from-scratch forward pass with Hugging Face Transformers (`requirements-dev.txt`); `record_demo.py` re-records the demo with Playwright.

Environment: Python 3.12, PyTorch 2.8.0 CPU, FP32, seed 42; HP EliteBook 840 G8 (Intel i5-1145G7, 4 cores / 8 threads, 15.7 GiB RAM), no GPU, no paid service. Thread count changes floating-point summation order, so retraining reproduces the protocol, not bit-identical epochs.

## Code map

| File | Purpose |
|---|---|
| `prepare.py` | Reads the parquet, builds the seed-42 patient split, keeps only note/type/criterion as inputs |
| `model.py` | From-scratch GPT-NeoX/Pythia blocks (RoPE, parallel residual), restricted 107-symbol head, constrained grammar, cached/uncached greedy decoding |
| `train.py` | Answer-only cross-entropy with class-balanced label and evidence weighting; validation-based checkpoint selection |
| `evaluate.py` | Calibration, freeze, paired base/adapted/baseline scoring, all metrics, error records |
| `inference.py` | Greedy speculative decoding with cache trimming; reference and tiled online-softmax attention |
| `distill.py` | Sequence-level distillation of the 14M draft |
| `benchmark.py` | Warm-up + median timing, RSS sampling, token-equality checks, attention benchmarks |
| `diagnose.py` | Teacher-forced label/evidence AUC diagnostics |
| `verify.py` | Split/leakage, cache, greedy, speculative, attention and app acceptance tests |
| `app.py`, `index.html` | Local web server and single-page interface |

## What `results/` contains

`deployment.json` (frozen target/drafts/bias), `freeze.json` (checkpoint SHA-256 written before test scoring), `train_config.json`, `history.json`, `selected.json`, `calibration.json`, `distill_log.json`, `experiments_summary.json` (every run, including failed ones), `{validation,test}_{majority,lead,base,adapted0,adapted}_{predictions,metrics}.json` (prediction *i* corresponds to line *i* of the split's JSONL; metrics include per-class P/R/F1, confusion matrices, inclusion/exclusion breakdowns and micro/macro evidence scores), `metrics_summary.json`, `errors.json`, `diagnostics.json`, `benchmark_log.json`, `reference_audit.json`, `verify_log.json`, and `first_submission/` (the original Pythia-14M checkpoint and outputs, kept for comparison).

## Data, models and licences

- Dataset: NCBI TrialGPT Criterion Annotations (public domain), https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations. GPT-4 prediction columns are never used as inputs.
- Models: EleutherAI Pythia-14M (rev. `cf967c0a`) and Pythia-70M (rev. `a39f36b1`), Apache-2.0; licence in `base/` and `base70/`.

## Submission links

- Repository and source: https://github.com/AbdullahMasood73/genai-assignment-2
- LaTeX source: https://github.com/AbdullahMasood73/genai-assignment-2/tree/main/report
- Adapted weights: https://github.com/AbdullahMasood73/genai-assignment-2/raw/refs/heads/main/results/adaptation.pt
- Demonstration: https://github.com/AbdullahMasood73/genai-assignment-2/blob/main/demo.mp4

The repository is private; the owner must grant the evaluator access for these links to work. All deliverables are also in the submission ZIP.

## AI use

OpenAI Codex assisted with the first version (planning, code, experiments, first report and demo script). Anthropic Claude (Claude Code) audited it and implemented the revision: configuration-driven model code, Pythia-70M adaptation, calibration, draft distillation, baselines, extended tests and benchmarks, the revised application and the revised report. All code was executed and tested; every reported number is produced by the included scripts.
