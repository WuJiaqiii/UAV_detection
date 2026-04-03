import torch
import torch.nn as nn

try:
    from torchvision.models import (
        resnet18,
        resnet34,
        mobilenet_v3_small,
        ResNet18_Weights,
        ResNet34_Weights,
        MobileNet_V3_Small_Weights,
    )
except Exception as e:
    raise ImportError(
        "torchvision is required for MaskImageClassifier. "
        f"Original error: {e}"
    )


class MaskImageClassifier(nn.Module):
    def __init__(
        self,
        backbone="resnet18",
        num_classes=8,
        in_chans=1,
        pretrained=True,
        dropout=0.0,
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.num_classes = int(num_classes)
        self.in_chans = int(in_chans)

        if self.backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
            feat_dim = model.fc.in_features
            self._adapt_first_conv(model, in_chans)
            model.fc = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(feat_dim, self.num_classes),
            )
            self.net = model

        elif self.backbone_name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            model = resnet34(weights=weights)
            feat_dim = model.fc.in_features
            self._adapt_first_conv(model, in_chans)
            model.fc = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(feat_dim, self.num_classes),
            )
            self.net = model

        elif self.backbone_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = mobilenet_v3_small(weights=weights)
            self._adapt_first_conv_mobilenet(model, in_chans)
            last_in = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(last_in, self.num_classes)
            self.net = model

        else:
            raise ValueError(
                f"Unsupported backbone={backbone}. "
                f"Choose from ['resnet18', 'resnet34', 'mobilenet_v3_small']"
            )

        if freeze_backbone:
            self.freeze_backbone()

    def _adapt_first_conv(self, model, in_chans):
        old_conv = model.conv1
        if old_conv.in_channels == in_chans:
            return

        new_conv = nn.Conv2d(
            in_chans,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        with torch.no_grad():
            if in_chans == 1:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                if in_chans < 3:
                    new_conv.weight[:, :in_chans].copy_(old_conv.weight[:, :in_chans])
                else:
                    new_conv.weight[:, :3].copy_(old_conv.weight)
                    for c in range(3, in_chans):
                        new_conv.weight[:, c:c+1].copy_(old_conv.weight.mean(dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        model.conv1 = new_conv

    def _adapt_first_conv_mobilenet(self, model, in_chans):
        first = model.features[0][0]  # Conv2dNormActivation -> first conv
        if first.in_channels == in_chans:
            return

        new_conv = nn.Conv2d(
            in_chans,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=(first.bias is not None),
        )

        with torch.no_grad():
            if in_chans == 1:
                new_conv.weight.copy_(first.weight.mean(dim=1, keepdim=True))
            else:
                if in_chans < 3:
                    new_conv.weight[:, :in_chans].copy_(first.weight[:, :in_chans])
                else:
                    new_conv.weight[:, :3].copy_(first.weight)
                    for c in range(3, in_chans):
                        new_conv.weight[:, c:c+1].copy_(first.weight.mean(dim=1, keepdim=True))
            if first.bias is not None:
                new_conv.bias.copy_(first.bias)

        model.features[0][0] = new_conv

    def freeze_backbone(self):
        for n, p in self.net.named_parameters():
            p.requires_grad = False

        if self.backbone_name.startswith("resnet"):
            for p in self.net.fc.parameters():
                p.requires_grad = True
        else:
            for p in self.net.classifier.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.net(x)