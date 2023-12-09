from turtle import forward
import torch
import torch.nn as nn
from torch.nn import functional as F

def create_model():
    pass

class U_net(nn.Module):
    def __init__(self, input_channel, device):
        super().__init__()
        self.i_channel = input_channel
        self.device = device
        self.Conv_Down = nn.ModuleList(
            [U_Conv(self.i_channel, 64, dense_Conv=[1, 2], outway='down'),
            U_Conv(64, 128, outway='down'),
            U_Conv(128, 256, outway='down'),
            U_Conv(256, 512, outway='down')]
        ).to(self.device)
        self.Conv_Up = nn.ModuleList(
            [U_Conv(512, 1024, outway='up'),
            U_Conv(1024, 512, outway='up'),
            U_Conv(512, 256, outway='up'),
            U_Conv(256, 128, outway='up')]
        ).to(self.device)
        self.Conv_out = U_Conv(128, 64, outway='out').to(self.device)
    
    def forward(self, x):
        down_out = []
        for conv in self.Conv_Down:
            out_d, x = conv(x)
            down_out.append(out_d)
        for i, conv in enumerate(self.Conv_Up):
            _, x = conv(x)
            x = torch.cat((down_out[3-i], x), 0)
        _, x = self.Conv_out(x)
        return(x)

class U_Conv(nn.Module):
    def __init__(self, input_channel, output_channel, dense_Conv=[], outway='down'):
        super().__init__()
        self.i_channel = input_channel
        self.o_channel = output_channel
        self.outway = outway
        num_Dense_Convs = 4
        # 卷积层，池化方式由层决定
        # self.U_Conv = nn.Sequential(
        #     nn.Conv2d(self.i_channel, self.o_channel, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=self.o_channel, out_channels=self.o_channel, kernel_size=3, padding=1),
        #     nn.ReLU()
        # )
        self.U_Convs = nn.Sequential()
        if 1 in dense_Conv:
            self.U_Convs.append(Dense_Block(self.i_channel, self.o_channel, num_Dense_Convs))
        else:
            self.U_Convs.append(nn.Conv2d(self.i_channel, self.o_channel, kernel_size=3, padding=1))
        self.U_Convs.append(nn.ReLU())
        if 2 in dense_Conv:
            self.U_Convs.append(Dense_Block(self.o_channel, self.o_channel, num_Dense_Convs))
        else:
            self.U_Convs.append(nn.Conv2d(self.o_channel, self.o_channel, kernel_size=3, padding=1))
        self.U_Convs.append(nn.ReLU())

        if self.outway == 'down':
            self.out_Conv = (nn.MaxPool2d(kernel_size=2, stride=2))
        elif self.outway == 'up':
            self.out_Conv = (nn.ConvTranspose2d(in_channels=self.o_channel, out_channels=self.o_channel//2, kernel_size=2, stride=2))
        elif self.outway == 'out':
            self.out_Conv = (nn.Conv2d(in_channels=self.o_channel, out_channels=2, kernel_size=1))
    
    def forward(self, x):
        x = self.U_Convs(x)
        out_d = x
        x = self.out_Conv(x)
        return out_d, x # 由于需要用到Down时的输出，故将其一并输出，但在Up时用不到，故用_接收

class Dense_Block(nn.Module):
    def __init__(self, input_c, output_c, num_convs):
        super().__init__()
        layer = []
        for i in range(num_convs):
            layer.append(Dense_Conv(input_c + output_c*i, output_c))
        self.net = nn.Sequential(*layer)
        self.trans = trans(input_c + output_c*num_convs, output_c)
    def forward(self, x):
        for conv in self.net:
            Y = conv(x)
            x = torch.cat((x, Y), dim=0) # 每次都将x与y连接
        x = self.trans(x)
        return x

def Dense_Conv(input_channels, output_channels):
    return nn.Sequential(
        #nn.BatchNorm2d(input_channels), 
        nn.ReLU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
    )

def trans(input_channels, output_channels):
    return nn.Sequential(
        #nn.BatchNorm2d(input_channels), 
        nn.ReLU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=1)
    )

if __name__ == '__main__':
    data = torch.randn((3, 1440, 994)).to('cuda')
    model = U_net(1, 'cuda')
    data = model(data)
    print(data.shape)