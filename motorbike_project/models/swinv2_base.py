from torchvision.models import swin_v2_b
import torch.nn as nn
import torch


class SwinV2Base(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.swin_v2_b = swin_v2_b(pretrained=True)  # Weights here are pretrained on ImageNet
        self.swin_v2_b.head = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.swin_v2_b(x)


if __name__ == '__main__':
    model = SwinV2Base()
    print(model)
    x = torch.randn(1, 3, 224, 224)  # Params: (batch_size, channels, height, width)
    y = model(x)
    print(y.shape)
