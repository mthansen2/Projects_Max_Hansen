import torchvision.models as models
import torch.nn as nn

def build_model(num_classes: int, arch: str = "resnet18", pretrained: bool = False):
    if arch == "resnet18":
        m = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
    elif arch == "resnet34":
        m = models.resnet34(weights=None if not pretrained else models.ResNet34_Weights.DEFAULT)
    elif arch == "resnet50":
        m = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ...
# in main(), after args parsed:
model = build_model(num_classes=len(CLASSES), arch="resnet18", pretrained=False)  # <-- set arch correctly

ckpt = torch.load(args.ckpt, map_location="cpu")

# ckpt might be:
# 1) full model (nn.Module)
# 2) dict with "model" (state_dict or module)
# 3) raw state_dict (OrderedDict)
if isinstance(ckpt, torch.nn.Module):
    model = ckpt
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"], strict=True)
elif isinstance(ckpt, dict) and "model" in ckpt:
    if isinstance(ckpt["model"], torch.nn.Module):
        model = ckpt["model"]
    else:
        model.load_state_dict(ckpt["model"], strict=True)
else:
    # raw OrderedDict state_dict
    model.load_state_dict(ckpt, strict=True)

model.to(device).eval()
