"""Recover the upstream base checkpoint when using the compact source repository."""
import hashlib
import urllib.request
from pathlib import Path

base=Path(__file__).resolve().parent/"base"
target=base/"model.safetensors"
expected="116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1"
if not target.exists():
    revision=(base/"revision.txt").read_text().strip()
    temporary=target.with_suffix(".download")
    urllib.request.urlretrieve(f"https://huggingface.co/EleutherAI/pythia-14m/resolve/{revision}/model.safetensors",temporary)
    if hashlib.sha256(temporary.read_bytes()).hexdigest()!=expected:
        raise ValueError("Upstream model checksum mismatch")
    temporary.replace(target)
if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
    raise ValueError("Base model checksum mismatch")
print("Base weights verified.")
