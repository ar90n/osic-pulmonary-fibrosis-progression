from itertools import islice
from pathlib import Path

import timm
import torch
from torch import nn
from torchtoolbox.nn import Swish

from .lightning import LightningModelBase
from .util import get_device


def create_ef_lowlevel_features_model(model: LightningModelBase, output_layer: int = 5):
    class _Model(nn.Module):
        def __init__(self, backbone, n: int):
            super().__init__()
            self._layers = backbone.as_sequential()[:n]

        def forward(self, inputs):
            x, _ = inputs
            for l in self._layers:
                x = l(x)
            return x.flatten(start_dim=1)

    return _Model(model._backbone, output_layer)


def _build_en_mlp_class(model_level: int):
    pretrained_model_name = [
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2a",
        "efficientnet_b3",
        "tf_efficientnet_b4_ns",
        "tf_efficientnet_b5_ns",
        "tf_efficientnet_b6_ns",
    ][model_level]

    class _c(LightningModelBase):
        name: str = f"en_b{model_level}_mlp"

        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self._backbone = timm.create_model(
                pretrained_model_name, num_classes=512, pretrained=True
            )
            self._meta = nn.Sequential(
                nn.Linear(9, 512),
                nn.BatchNorm1d(512),
                Swish(),
                nn.Dropout(p=0.2),
                nn.Linear(512, 256),  # FC layer output will have 256 features
                nn.BatchNorm1d(256),
                Swish(),
                nn.Dropout(p=0.2),
            )
            self._classifier = nn.Linear(512 + 256, 1)

        def backbone(self, x):
            return self._backbone(x)

        def meta(self, x):
            return self._meta(x)

        def features(self, inputs):
            x, y = inputs
            x = self.backbone(x)
            y = self.meta(y)
            return torch.cat((x, y), dim=1)

        def classify(self, features):
            output = self._classifier(features)
            return output.view(output.size(0), -1)

        def forward(self, inputs):
            features = self.features(inputs)
            return self.classify(features)

    return _c


EfficientNetB0MLP = _build_en_mlp_class(0)
EfficientNetB1MLP = _build_en_mlp_class(1)
EfficientNetB2MLP = _build_en_mlp_class(2)
EfficientNetB3MLP = _build_en_mlp_class(3)
EfficientNetB4MLP = _build_en_mlp_class(4)
EfficientNetB5MLP = _build_en_mlp_class(5)
EfficientNetB6MLP = _build_en_mlp_class(6)


def get_model_class(model_name):
    return {
        EfficientNetB0MLP.name: EfficientNetB0MLP,
        EfficientNetB1MLP.name: EfficientNetB1MLP,
        EfficientNetB2MLP.name: EfficientNetB2MLP,
        EfficientNetB3MLP.name: EfficientNetB3MLP,
        EfficientNetB4MLP.name: EfficientNetB4MLP,
        EfficientNetB5MLP.name: EfficientNetB5MLP,
        EfficientNetB6MLP.name: EfficientNetB6MLP,
    }[model_name]


def load_from_checkpoint(model_name, path_to_cktp: Path) -> LightningModelBase:
    return get_model_class(model_name).load_from_checkpoint(str(path_to_cktp))
