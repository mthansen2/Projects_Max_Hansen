# eval_kaggle_test28.py
# ------------------------------------------------------------
# Evaluate on Kaggle ASL Alphabet "test" folder (one image per class ~28 images).
# This is a sanity check (not a statistically meaningful test set).
#
# Works with checkpoints saved as:
#   - raw state_dict (OrderedDict)
#   - dict with "state_dict"
#   - dict with "model" (state_dict or nn.Module)
#   - full nn.Module saved directly
#
# Usage (PowerShell, one line):
#   python .\code_base\src\eval_kaggle_test28.py --test-dir ".\asl_alphabet_test" --ckpt ".\code_base\checkpoints\best.pt" --device cpu --arch resnet18
# ------------------------------------------------------------

import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.models as models

# Kaggle ASL Alphabet classes (common order)
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
NAME_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def label_from_name(p: Path) -> int:
    """
    Kaggle test files look like:
      A_test.jpg, ..., Z_test.jpg, del_test.jpg, nothing_test.jpg, space_test.jpg
    """
    stem = p.stem.lower()
    if stem.startswith("nothing"):
        return NAME_TO_IDX["nothing"]
    if stem.startswith("space"):
        return NAME_TO_IDX["space"]
    if stem.startswith("del"):
        return NAME_TO_IDX["del"]
    # A_test -> 'a' -> A
    return NAME_TO_IDX[stem[0].upper()]


def strip_module_prefix(state_dict: dict) -> dict:
    """If keys start with 'module.' (DDP), strip that prefix."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "", 1)] = v
    return new_sd


def build_model(num_classes: int, arch: str, pretrained: bool) -> nn.Module:
    """
    Recreate a ResNet classifier so we can load a state_dict.
    Set arch to what you trained (resnet18/34/50).
    """
    weights = None
    if pretrained:
        if arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
        elif arch == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT
        elif arch == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT

    if arch == "resnet18":
        m = models.resnet18(weights=weights)
    elif arch == "resnet34":
        m = models.resnet34(weights=weights)
    elif arch == "resnet50":
        m = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_checkpoint_into_model(model: nn.Module, ckpt_path: str) -> nn.Module:
    """
    Supports:
      - full nn.Module saved directly
      - dict with 'model' (state_dict or module)
      - dict with 'state_dict'
      - raw state_dict (OrderedDict)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Case 1: full model object
    if isinstance(ckpt, nn.Module):
        return ckpt

    # Case 2: dict container
    if isinstance(ckpt, dict):
        # "model" may be a module or state_dict
        if "model" in ckpt:
            if isinstance(ckpt["model"], nn.Module):
                return ckpt["model"]
            if isinstance(ckpt["model"], (dict, OrderedDict)):
                sd = strip_module_prefix(ckpt["model"])
                model.load_state_dict(sd, strict=True)
                return model

        # common key: "state_dict"
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], (dict, OrderedDict)):
            sd = strip_module_prefix(ckpt["state_dict"])
            model.load_state_dict(sd, strict=True)
            return model

        # other possible keys
        for key in ("net", "weights", "params"):
            if key in ckpt and isinstance(ckpt[key], (dict, OrderedDict)):
                sd = strip_module_prefix(ckpt[key])
                model.load_state_dict(sd, strict=True)
                return model

    # Case 3: raw state_dict
    if isinstance(ckpt, (dict, OrderedDict)):
        sd = strip_module_prefix(ckpt)
        model.load_state_dict(sd, strict=True)
        return model

    raise TypeError(f"Unrecognized checkpoint type: {type(ckpt)}")


@torch.no_grad()
def eval_folder(model: nn.Module, test_dir: str, device: torch.device, img_size: int, no_imagenet_norm: bool):
    """
    Evaluate a flat folder of .jpg files (Kaggle test28 style).
    """
    tfm_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    # If you trained with ImageNet normalization (typical when using pretrained ResNet),
    # keep it. If you trained without it, use --no-imagenet-norm.
    if not no_imagenet_norm:
        tfm_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )
    tfm = transforms.Compose(tfm_list)

    paths = sorted(Path(test_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No .jpg files found in: {test_dir}")

    total = 0
    correct1 = 0
    correct3 = 0

    print("\nPer-image results:")
    print("-" * 90)

    for p in paths:
        y = label_from_name(p)
        x = tfm(Image.open(p).convert("RGB")).unsqueeze(0).to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        pred1 = int(probs.argmax(dim=1).item())
        top3 = probs.topk(k=3, dim=1).indices.squeeze(0).tolist()
        conf1 = float(probs[0, pred1].item())

        correct1 += int(pred1 == y)
        correct3 += int(y in top3)
        total += 1

        top3_str = ", ".join([CLASSES[i] for i in top3])
        print(
            f"{p.name:20s}  gt={CLASSES[y]:7s}  pred={CLASSES[pred1]:7s}  "
            f"conf={conf1:0.3f}  top3=[{top3_str}]"
        )

    acc1 = correct1 / max(1, total)
    acc3 = correct3 / max(1, total)

    print("-" * 90)
    print(f"Total: {total}")
    print(f"Top-1 accuracy: {acc1:.4f} ({correct1}/{total})")
    print(f"Top-3 accuracy: {acc3:.4f} ({correct3}/{total})\n")


def main():
    ap = argparse.ArgumentParser(description="Evaluate Kaggle ASL Alphabet test28 folder.")
    ap.add_argument("--test-dir", required=True, help="Folder containing flat .jpg files (e.g., A_test.jpg, ...).")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt).")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--pretrained", action="store_true",
                    help="Build model with ImageNet pretrained backbone (ONLY if you trained that way).")
    ap.add_argument("--no-imagenet-norm", action="store_true",
                    help="Disable ImageNet normalization (use if you trained without it).")
    args = ap.parse_args()

    # Respect user choice, but fall back safely if CUDA isn't available
    if args.device != "cpu" and not torch.cuda.is_available():
        print("[warn] CUDA not available, using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = build_model(num_classes=len(CLASSES), arch=args.arch, pretrained=args.pretrained)
    model = load_checkpoint_into_model(model, args.ckpt)
    model.to(device).eval()

    eval_folder(model, args.test_dir, device, args.img_size, args.no_imagenet_norm)


if __name__ == "__main__":
    main()
