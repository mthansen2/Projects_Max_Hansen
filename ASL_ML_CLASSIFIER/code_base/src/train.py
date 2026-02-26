from __future__ import annotations
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from .data.dataset import make_loaders
from .models.factory import build_model
from .utils import ensure_dir


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.size(0)

    return loss_sum / max(1, total), correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/raw")
    ap.add_argument("--split-dir", type=str, default="data/splits")
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.ckpt_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device} pretrained={args.pretrained}")

    train_loader, val_loader = make_loaders(
        Path(args.data_root),
        Path(args.split_dir),
        args.img_size,
        args.batch_size,
        args.num_workers,
    )

    model = build_model(pretrained=args.pretrained).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp and device == "cuda"))
    ce = nn.CrossEntropyLoss()

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")

        run_loss = 0.0
        run_correct = 0
        run_total = 0

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(not args.no_amp and device == "cuda")):
                logits = model(x)
                loss = ce(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            run_correct += (pred == y).sum().item()
            run_total += y.numel()

            pbar.set_postfix(
                loss=run_loss / max(1, run_total),
                acc=run_correct / max(1, run_total),
            )

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[val] loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = Path(args.ckpt_dir) / "best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[ckpt] saved {ckpt_path} (best_acc={best_acc:.4f})")

    print(f"[done] best_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
