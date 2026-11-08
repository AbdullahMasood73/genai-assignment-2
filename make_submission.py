"""Build I230756_GenAI_A2_Submission.zip from the project folder (excludes environments and caches)."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "I230756_GenAI_A2_Submission.zip"
INCLUDE_DIRS = ["base", "base70", "data", "report", "results", "experiments"]
TOP_FILES = [".gitignore", "README.md", "Start-Screening.cmd", "requirements.txt", "requirements-dev.txt",
             "app.py", "index.html", "model.py", "prepare.py", "train.py", "evaluate.py", "inference.py",
             "benchmark.py", "distill.py", "diagnose.py", "verify.py", "setup_model.py", "audit_reference.py",
             "record_demo.py", "make_submission.py", "demo.mp4", "train-00000-of-00001.parquet", "GenAI_Assignment#2.pdf"]
SKIP_SUFFIX = {".pyc", ".download", ".log~"}
SKIP_NAMES = {"app_smoke.log", "app_final.log", "record.log", "paper.log"}


def wanted(path):
    if "__pycache__" in path.parts or path.suffix in SKIP_SUFFIX or path.name in SKIP_NAMES:
        return False
    if path.parts[0] == "experiments" and path.suffix == ".pt":
        return False          # experiment checkpoints are not needed; the frozen one is in results/
    return True


files = [ROOT / f for f in TOP_FILES]
for d in INCLUDE_DIRS:
    files += sorted(p for p in (ROOT / d).rglob("*") if p.is_file() and wanted(p.relative_to(ROOT)))
missing = [str(f) for f in files if not f.exists()]
assert not missing, missing
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in files:
        z.write(f, f.relative_to(ROOT).as_posix())
print(f"{OUT.name}: {len(files)} files, {OUT.stat().st_size/2**20:.1f} MiB")
