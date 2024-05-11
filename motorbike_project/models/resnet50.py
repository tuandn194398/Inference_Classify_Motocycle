import torch
import torch.nn as nn
from torchvision.models import resnet50


class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.resnet50 = resnet50(pretrained=True)  # Weights here are pretrained on ImageNet
        self.resnet50.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.resnet50(x)


if __name__ == '__main__':
    model = ResNet50()
    print(model)
    x = torch.randn(1, 3, 224, 224)  # Params: (batch_size, channels, height, width)
    y = model(x)
    print(y.shape)
