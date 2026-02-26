from __future__ import annotations
import argparse
import random
from pathlib import Path

from src.labels import CLASSES
from src.utils import find_train_dir, ensure_dir

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/raw")
    ap.add_argument("--out-dir", type=str, default="data/splits")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = ensure_dir(args.out_dir)

    train_dir = find_train_dir(data_root)

    train_lines: list[str] = []
    val_lines: list[str] = []

    for cls in CLASSES:
        cls_dir = train_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")

        imgs = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if len(imgs) == 0:
            raise RuntimeError(f"No images found for class {cls} in {cls_dir}")

        random.shuffle(imgs)
        n_val = int(len(imgs) * args.val_frac)

        val_set = imgs[:n_val]
        train_set = imgs[n_val:]

        for p in train_set:
            rel = p.relative_to(train_dir)
            train_lines.append(f"{rel.as_posix()} {cls}\n")

        for p in val_set:
            rel = p.relative_to(train_dir)
            val_lines.append(f"{rel.as_posix()} {cls}\n")

    random.shuffle(train_lines)
    random.shuffle(val_lines)

    (out_dir / "train.txt").write_text("".join(train_lines), encoding="utf-8")
    (out_dir / "val.txt").write_text("".join(val_lines), encoding="utf-8")

    print(f"[ok] wrote {out_dir/'train.txt'} ({len(train_lines)} samples)")
    print(f"[ok] wrote {out_dir/'val.txt'} ({len(val_lines)} samples)")
    print(f"[info] train_dir = {train_dir}")


if __name__ == "__main__":
    main()
