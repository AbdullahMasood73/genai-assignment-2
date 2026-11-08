"""Prepare the official patient-grouped benchmark from the published parquet."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "train-00000-of-00001.parquet"
COLUMNS = ["annotation_id", "patient_id", "note", "criterion_type", "criterion_text",
           "expert_eligibility", "expert_sentences"]


def sentences(note):
    matches = list(re.finditer(r"(?m)^\s*(\d+)\.\s+", note))
    return {int(m[1]): note[m.end():matches[i+1].start() if i+1 < len(matches) else len(note)].strip()
            for i, m in enumerate(matches)}


def prepare():
    rows = pq.read_table(SOURCE, columns=COLUMNS).to_pylist()
    ids = sorted({row["patient_id"] for row in rows})
    assert len(ids) == 53 and len({row["annotation_id"] for row in rows}) == len(rows)
    np.random.default_rng(42).shuffle(ids)
    manifest = {"seed": 42, "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "splits": {}}
    output = ROOT / "data"
    output.mkdir(exist_ok=True)
    for name, patients in [("train", ids[:37]), ("validation", ids[37:45]), ("test", ids[45:])]:
        items = []
        for row in rows:
            if row["patient_id"] not in patients:
                continue
            evidence = json.loads(row["expert_sentences"])
            lookup = sentences(row["note"])
            assert set(evidence) <= lookup.keys(), (row["annotation_id"], evidence, list(lookup))
            items.append({"id": row["annotation_id"], "patient": row["patient_id"],
                          "input": {"note": row["note"], "type": row["criterion_type"], "criterion": row["criterion_text"]},
                          "target": {"label": row["expert_eligibility"], "evidence": evidence}})
        (output / (name + ".jsonl")).write_text("".join(json.dumps(x) + "\n" for x in items), encoding="utf-8")
        manifest["splits"][name] = {"patients": patients, "records": len(items)}
    assert sum(v["records"] for v in manifest["splits"].values()) == len(rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({name: v["records"] for name, v in manifest["splits"].items()})


if __name__ == "__main__":
    prepare()
