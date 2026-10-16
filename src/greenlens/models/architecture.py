import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes: int = 15, pretrained: bool = True):
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = resnet18(weights=weights)

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    return model