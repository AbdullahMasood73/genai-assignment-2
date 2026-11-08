"""Download (if absent) and SHA-256-verify the pinned upstream Pythia checkpoints."""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PINNED={
    "base":("EleutherAI/pythia-14m","116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1"),
    "base70":("EleutherAI/pythia-70m","ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15"),
}


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


for folder,(repo,expected) in PINNED.items():
    base=ROOT/folder
    target=base/"model.safetensors"
    if not target.exists():
        revision=(base/"revision.txt").read_text().strip()
        url=f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors"
        print(f"Downloading {repo}@{revision[:10]} ...",flush=True)
        temporary=target.with_suffix(".download")
        try:
            urllib.request.urlretrieve(url,temporary)
        except OSError as e:
            sys.exit(f"Download failed ({e}). Check the internet connection, or copy {folder}/model.safetensors from the submission ZIP.")
        if digest(temporary)!=expected:
            temporary.unlink()
            sys.exit(f"Upstream checksum mismatch for {repo}.")
        temporary.replace(target)
    if digest(target)!=expected:
        sys.exit(f"Checksum mismatch for {target}. Delete it and run again to re-download.")
    print(f"{repo} weights verified.")
