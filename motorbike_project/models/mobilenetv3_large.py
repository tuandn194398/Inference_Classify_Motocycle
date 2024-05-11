import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.mobilenet_v3_large = mobilenet_v3_large(pretrained=True)  # Weights here are pretrained on ImageNet
        self.mobilenet_v3_large.classifier = nn.Linear(960, num_classes)

    def forward(self, x):
        return self.mobilenet_v3_large(x)


if __name__ == '__main__':
    model = MobileNetV3Large()
    print(model)
    x = torch.randn(1, 3, 224, 224)  # Params: (batch_size, channels, height, width)
    y = model(x)
    print(y.shape)
