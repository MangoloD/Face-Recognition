import torch
import torch.nn as nn


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # 10*10*3
            nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 5*5*10

            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # 3*3*16
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # 1*1*32
            nn.BatchNorm2d(32),
            nn.PReLU()
        )

        self.conv_con = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv_loc = nn.Conv2d(32, 4, kernel_size=1, stride=1)

    def forward(self, input_x):
        output_y = self.pre_layer(input_x)
        con = torch.sigmoid(self.conv_con(output_y))
        loc = self.conv_loc(output_y)
        return con, loc


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # 22*22*28
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 11*11*28

            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # 9*9*48
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 4*4*48

            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # 3*3*64
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.fc = nn.Linear(64 * 3 * 3, 128)
        self.relu_ = nn.PReLU()

        self.con = nn.Linear(128, 1)
        self.loc = nn.Linear(128, 14)

    def forward(self, input_x):
        output_y = self.pre_layer(input_x)
        output_y = output_y.view(output_y.size(0), -1)
        output_y = self.fc(output_y)
        output_y = self.relu_(output_y)

        con = torch.sigmoid(self.con(output_y))
        loc = self.loc(output_y)

        return con, loc


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # 46*46*32
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 23*23*32

            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # 21*21*64
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 10*10*64

            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 8*8*64
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4*4*64

            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )

        self.fc = nn.Linear(128 * 3 * 3, 256)
        self.relu_ = nn.PReLU()

        self.con = nn.Linear(256, 1)
        self.loc = nn.Linear(256, 14)

    def forward(self, input_x):
        output_y = self.pre_layer(input_x)
        output_y = output_y.view(output_y.size(0), -1)
        output_y = self.fc(output_y)
        output_y = self.relu_(output_y)

        con = torch.sigmoid(self.con(output_y))
        loc = self.loc(output_y)

        return con, loc


if __name__ == '__main__':
    p_net = PNet()
    r_net = RNet()
    o_net = ONet()
    px = torch.randn([10, 3, 12, 12])
    rx = torch.randn([10, 3, 24, 24])
    ox = torch.randn([10, 3, 48, 48])
    print(p_net(px)[0].shape)
    print(p_net(px)[1].shape)
    print(r_net(rx)[0].shape)
    print(r_net(rx)[1].shape)
    print(o_net(ox)[0].shape)
    print(o_net(ox)[1].shape)
