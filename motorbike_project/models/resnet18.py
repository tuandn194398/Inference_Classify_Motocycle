import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.resnet18 = resnet18(pretrained=True)  # Weights here are pretrained on ImageNet
        self.resnet18.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet18(x)


if __name__ == '__main__':
    model = ResNet18()
    print(model)
    x = torch.randn(1, 3, 224, 224)  # Params: (batch_size, channels, height, width)
    y = model(x)
    print(y.shape)
