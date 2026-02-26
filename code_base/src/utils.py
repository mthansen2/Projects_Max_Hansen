from __future__ import annotations
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_train_dir(data_root: str | Path) -> Path:
    """
    Returns the directory that directly contains the class folders:
      A..Z, del, nothing, space

    Handles Kaggle nesting like:
      data_root/asl_alphabet_train/asl_alphabet_train/<CLASS>/*.jpg
    """
    data_root = Path(data_root)

    required = {"A", "B", "C", "X", "Y", "Z", "del", "nothing", "space"}

    def looks_like_train_dir(p: Path) -> bool:
        if not (p.exists() and p.is_dir()):
            return False
        names = {c.name for c in p.iterdir() if c.is_dir()}
        return required.issubset(names)

    # Common locations
    candidates = [
        data_root / "asl_alphabet_train" / "asl_alphabet_train",
        data_root / "asl_alphabet_train",
        data_root,
    ]
    for c in candidates:
        if looks_like_train_dir(c):
            return c

    # Small recursive scan (covers weird unzips)
    for p in data_root.rglob("*"):
        if p.is_dir() and looks_like_train_dir(p):
            return p

    raise FileNotFoundError(
        f"Could not find training directory under {data_root}. "
        "Expected a folder containing class dirs like A, B, ..., space, del, nothing."
    )
