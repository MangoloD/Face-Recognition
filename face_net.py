import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.models as models

torch.manual_seed(0)


class ArcSoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super(ArcSoftmax, self).__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)), requires_grad=True)

    def forward(self, x, s=64, m=0.5):
        x_normal = f.normalize(x, dim=1)
        w_normal = f.normalize(self.w, dim=0)

        cos_a = torch.matmul(x_normal, w_normal) / s
        a = torch.acos(cos_a)

        arc_softmax = torch.exp(s * torch.cos(a + m)) / (
                torch.sum(torch.exp(s * cos_a), dim=1, keepdim=True)
                - torch.exp(s * cos_a) + torch.exp(s * torch.cos(a + m))
        )

        return arc_softmax


class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.feature_net = models.shufflenet_v2_x1_0(True)
        self.feature = nn.Sequential(
            nn.Linear(1024, 512)
        )
        self.feature_net.fc = self.feature
        self.arc_softmax = ArcSoftmax(512, 105)

    def forward(self, x):
        feature = self.feature_net(x)
        return feature, self.arc_softmax(feature)

    def encode(self, x):
        return self.feature_net(x)


if __name__ == '__main__':
    print(models.shufflenet_v2_x1_0(True))
    net = FaceNet()
    print(net)
