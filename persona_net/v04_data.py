from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List


def _read_items(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("items", payload.get("training_items", [])))


def load_v04_split(root: str | Path, split: str) -> List[Dict]:
    """Load the public v0.4 phenotype split from its manifest.

    The public repository intentionally excludes the sealed terminal inputs and
    scoring key. This loader knows only train, validation, and adversarial.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    if split == "train":
        names = manifest["train_files"]
    elif split == "validation":
        names = manifest["validation_files"]
    elif split == "adversarial":
        names = [manifest["adversarial_file"]]
    else:
        raise ValueError("split must be train, validation, or adversarial")

    items: List[Dict] = []
    for name in names:
        items.extend(_read_items(root / name))

    expected = int(manifest["counts"][split])
    if len(items) != expected:
        raise ValueError(
            f"{split} count mismatch: manifest says {expected}, loaded {len(items)}"
        )
    return items


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
