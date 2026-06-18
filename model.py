import torch
import torch.nn as nn
from torchvision import models


class SEBlock(nn.Module):
    """
    SE 注意力模块：Squeeze-and-Excitation Block

    作用：
    1. Squeeze：通过全局平均池化压缩空间信息，得到每个通道的全局响应。
    2. Excitation：通过两层全连接网络学习每个通道的重要性权重。
    3. Scale：将学习到的通道权重乘回原特征图，从而增强重要通道，抑制不重要通道。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)

        return x * y


class ConvSEBlock(nn.Module):
    """
    卷积 + BN + ReLU + SE + 池化 的基本模块
    """

    def __init__(self, in_channels, out_channels, use_pool=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels)
        ]

        if use_pool:
            layers.append(nn.MaxPool2d(2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SmallCNN(nn.Module):
    """
    加入 SE 注意力机制的四层 CNN 模型

    输入：
        3 × 224 × 224 的 RGB 图像

    网络结构：
        ConvSEBlock 1: 3   -> 32
        ConvSEBlock 2: 32  -> 64
        ConvSEBlock 3: 64  -> 128
        ConvSEBlock 4: 128 -> 256
        Global Average Pooling
        Dropout
        Linear 分类层

    输出：
        2 类：
        0: no_glasses
        1: glasses
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            ConvSEBlock(3, 32, use_pool=True),
            ConvSEBlock(32, 64, use_pool=True),
            ConvSEBlock(64, 128, use_pool=True),
            ConvSEBlock(128, 256, use_pool=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_resnet18(num_classes=2, pretrained=False):
    """
    构建 ResNet18 模型。
    注意：这个分支没有额外加入 SE 注意力。
    如果要使用 SE 注意力模型，请在训练时选择 --model_name small_cnn。
    """

    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=True)
    else:
        try:
            model = models.resnet18(weights=None)
        except Exception:
            model = models.resnet18(pretrained=False)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_model(model_name="small_cnn", num_classes=2, pretrained=False):
    model_name = model_name.lower()

    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)

    if model_name == "small_cnn":
        return SmallCNN(num_classes=num_classes)

    if model_name == "cnn_se":
        return SmallCNN(num_classes=num_classes)

    raise ValueError("model_name 只能是 resnet18、small_cnn 或 cnn_se")


if __name__ == "__main__":
    model = build_model(model_name="small_cnn", num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print(model)
    print("输出形状：", y.shape)