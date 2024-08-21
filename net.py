# coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F


class RCAFM(nn.Module):
    def __init__(self, in_ch):
        super(RCAFM, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
            nn.LeakyReLU()
        )
        self.gap_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch*2, in_ch*2, kernel_size=3, padding=1)
        )
        self.conv1_sigmoid_1 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.conv1_sigmoid_2 = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x_s, x_d):
        x_s_1 = self.conv1_1(x_s)
        x_d_1 = self.conv1_2(x_d)
        x = torch.cat([x_s_1, x_d_1], dim=1)
        x = self.gap_conv(x)
        x_s_1 = x_s_1 * self.conv1_sigmoid_1(x) + x_s
        x_d_1 = x_d_1 * self.conv1_sigmoid_2(x) + x_d
        x = torch.cat([x_s_1, x_d_1], dim=1)
        return x


# 消融实验：RCAFM变成直接concat
# class RCAFM(nn.Module):
#     def __init__(self, in_ch):
#         super(RCAFM, self).__init__()
#
#     def forward(self, x_s, x_d):
#         x = torch.cat([x_s, x_d], dim=1)
#         return x


class RDBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RDBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch*3, in_ch, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_ch*4, out_ch, kernel_size=1, padding=0),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = torch.cat([x, self.conv1(x)], dim=1)
        x = torch.cat([x, self.conv2(x)], dim=1)
        x = torch.cat([x, self.conv3(x)], dim=1)
        x = self.down_conv(x)
        return x


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        ch = [1, 8, 16, 32, 64, 128]
        self.s_conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=1, padding=0),
            nn.LeakyReLU()
        )
        self.d_conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=1, padding=0),
            nn.LeakyReLU()
        )
        self.fusion1 = RCAFM(ch[1])
        self.s_downconv1 = nn.Sequential(
            nn.Conv2d(ch[1], ch[1], kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU()
        )
        self.d_downconv1 = nn.Sequential(
            nn.Conv2d(ch[1], ch[1], kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU()
        )
        self.s_conv2 = RDBlock(ch[1], ch[2])
        self.d_conv2 = RDBlock(ch[1], ch[2])
        self.fusion2 = RCAFM(ch[2])
        self.s_downconv2 = nn.Sequential(
            nn.Conv2d(ch[2], ch[2], kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU()
        )
        self.d_downconv2 = nn.Sequential(
            nn.Conv2d(ch[2], ch[2], kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU()
        )
        self.s_conv3 = RDBlock(ch[2], ch[3])
        self.d_conv3 = RDBlock(ch[2], ch[3])
        self.fusion3 = RCAFM(ch[3])

        self.decoder4 = nn.Sequential(
            RDBlock(ch[5], ch[4]),
            RDBlock(ch[4], ch[3])
        )
        self.decoder3 = nn.Sequential(
            RDBlock(ch[4], ch[3]),
            RDBlock(ch[3], ch[2])
        )
        self.decoder2 = nn.Sequential(
            RDBlock(ch[3], ch[2]),
            RDBlock(ch[2], ch[1])
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(ch[1], ch[0], kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, image_s, image_d):
        # encoder
        x_s = self.s_conv1(image_s)
        x_d = self.d_conv1(image_d)
        x_1 = self.fusion1(x_s, x_d)

        x_s = self.s_downconv1(x_s)
        x_d = self.d_downconv1(x_d)
        x_s = self.s_conv2(x_s)
        x_d = self.d_conv2(x_d)
        x_2 = self.fusion2(x_s, x_d)

        x_s = self.s_downconv2(x_s)
        x_d = self.d_downconv2(x_d)
        x_s = self.s_conv3(x_s)
        x_d = self.d_conv3(x_d)
        x_3 = self.fusion3(x_s, x_d)

        # decoder
        h, w = x_2.shape[2], x_2.shape[3]
        x = self.decoder4(torch.cat([x_3, x_s, x_d], dim=1))
        x = F.interpolate(x, (h, w))

        h, w = x_1.shape[2], x_1.shape[3]
        x = self.decoder3(torch.cat([x_2, x], dim=1))
        x = F.interpolate(x, (h, w))

        x = self.decoder2(torch.cat([x_1, x], dim=1))
        x = self.decoder1(x)
        return x / 2 + 0.5


if __name__ == '__main__':
    import time
    from torchsummary import summary

    ch_in = 1
    n = 50  # 循环次数
    x = torch.randn(1, ch_in, 480, 640).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionNet().to(device)
    model.eval()

    for i in range(5):  # 预先跑几个数据
        train_y = model(x, x)

    start_time = time.time()
    for i in range(n):
        train_y = model(x, x)
    running_time = time.time() - start_time

    print(summary(model, [(ch_in, 480, 640), (ch_in, 480, 640)]))

    print('Running time is {:.4f}s'.format(running_time / n))

