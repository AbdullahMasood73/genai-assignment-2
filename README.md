# Trial screening workbench

Muhammad Abdullah Masood · I230756 · Generative AI Assignment 2

## Start the application

Extract the entire submission ZIP to a writable folder on Windows. Install 64-bit Python 3.12 with its Windows launcher if needed. Double-click **Start-Screening.cmd**. On the first run the launcher installs dependencies in a local environment and verifies the base checkpoint. Allow several minutes and internet access for this step. When ready it opens http://127.0.0.1:8764 in your browser. Keep the launcher window open while using the app. Close it to stop the server. Later launches reuse the environment.

Load the example or enter a note and one criterion per line, prefixed by "inclusion |" or "exclusion |". Choose **Compare all three modes** to measure real inference on this request. **Export screening report** saves the result. The app runs on loopback, processes notes in memory, and requires no model API key. Use fictional records for demonstrations.

## Submission contents

- report/paper.pdf is the compiled IEEE paper. report/paper.tex, body.tex, tables.tex, IEEEtran.cls and the PNG figures are its source project. Build paper.tex with Tectonic from a shell or upload the report directory to a LaTeX editor.
- demo.mp4 is the captioned application recording. The submitted archive includes the complete video.
- results/adaptation.pt contains all trained parameters. The input embedding comes from the public base model. setup_model.py verifies its hash and downloads the pinned original if absent. The ZIP also includes the original weights.
- data/manifest.json records the patient groups and source checksum. The prepared JSONL files retain original annotation IDs and separate input fields from gold targets.
- results/ contains the validation history, frozen checkpoint hash, full prediction lists, overall/type-specific metrics, confusion matrices, all error records, and decoder and attention benchmarks. Prediction position matches the corresponding JSONL line, including before/after comparisons.

## Reproduce the experiments

Using the installed Python environment from the project directory, run these scripts in order. Training and evaluation commands are for reproducing research; the browser workflow uses the launcher.

    python setup_model.py
    python prepare.py
    python train.py
    python evaluate.py
    python benchmark.py
    python verify.py

Run retraining in a separate extracted copy to preserve the supplied final outputs. Python 3.12, CPU PyTorch, FP32, two Torch threads, seed 42. The original run used an HP EliteBook 840 G8 with 15.7 GiB RAM. No GPU or paid service was used. Training updates the decoder blocks, final normalization and restricted head. The first two original blocks are reused as a compatible draft. The reduced output alphabet is for structured screening only.

The held-out test result is 126 correct labels out of 206. The model predicted only not excluded and not enough information. It retrieved zero annotated sentences, so its micro evidence F1 is zero. Per-record macro evidence F1 credits pairs of empty sets. This educational system cannot determine clinical eligibility.

## Sources and redistribution

The dataset is the public-domain NCBI TrialGPT Criterion Annotations release. Pythia-14M by EleutherAI is Apache-2.0 licensed. Its license and revision are included under base/. Released base, tokenizer and configuration: https://huggingface.co/EleutherAI/pythia-14m . Dataset: https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations . Research citations and AI assistance are documented in the paper.

## Submission links

Repository and source project: https://github.com/AbdullahMasood73/genai-assignment-2

LaTeX source: https://github.com/AbdullahMasood73/genai-assignment-2/tree/main/report

Adaptation download: https://github.com/AbdullahMasood73/genai-assignment-2/raw/refs/heads/main/results/adaptation.pt

Demonstration: https://github.com/AbdullahMasood73/genai-assignment-2/blob/main/demo.mp4

The repository is private. The repository owner must grant the evaluator access for the hosted links to work. All four deliverables are also available in the extracted submission.
