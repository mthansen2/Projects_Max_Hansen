from __future__ import annotations
import torch
from torchvision import models

from ..labels import NUM_CLASSES
from .resnet_custom import resnet18 as resnet18_custom


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> torch.nn.Module:
    model = resnet18_custom(num_classes=num_classes)

    if pretrained:
        tv = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        sd = tv.state_dict()

        # Drop classifier (shape mismatch with 29 classes)
        sd.pop("fc.weight", None)
        sd.pop("fc.bias", None)

        missing, unexpected = model.load_state_dict(sd, strict=False)

        bad_missing = [k for k in missing if not k.startswith("fc.")]
        if bad_missing:
            raise RuntimeError(f"Unexpected missing keys: {bad_missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected}")

    return model
