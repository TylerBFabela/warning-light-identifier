"""Shared code: choosing a device, the two model designs, image preprocessing, and loading a saved model."""
import json
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms

MEAN = (0.5, 0.5, 0.5)   # pixel values are shifted/scaled to roughly -1..1 before the network sees them
STD = (0.5, 0.5, 0.5)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"          # the Apple GPU
    return "cpu"


class SmallCNN(nn.Module):
    """Plain convolutional network: 4 x (conv -> batchnorm -> relu -> pool), then average-pool and a linear layer."""

    def __init__(self, n_classes, width=32):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, width), block(width, width * 2), block(width * 2, width * 4), block(width * 4, width * 4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(width * 4, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class ResBlock(nn.Module):
    """Two convolutions plus a 'skip connection' that adds the block's input back to its output."""

    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        same_shape = stride == 1 and c_in == c_out
        self.skip = nn.Identity() if same_shape else nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + self.skip(x))


class SmallResNet(nn.Module):
    def __init__(self, n_classes, width=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.stages = nn.Sequential(
            ResBlock(width, width, 1),
            ResBlock(width, width * 2, 2),
            ResBlock(width * 2, width * 4, 2),
            ResBlock(width * 4, width * 4, 2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(width * 4, n_classes))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


def build_model(arch, n_classes):
    if arch == "small_cnn":
        return SmallCNN(n_classes)
    if arch == "small_resnet":
        return SmallResNet(n_classes)
    raise ValueError(f"unknown arch: {arch}")


def eval_transform(img):
    """How every image is prepared before the network sees it. Training and prediction MUST use the same steps."""
    return transforms.Compose([transforms.Resize((img, img)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_model(model_dir, device="cpu"):
    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text())
    classes = json.loads((d / "classes.json").read_text())
    model = build_model(cfg["arch"], len(classes))
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    return model.to(device).eval(), classes, cfg