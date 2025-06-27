import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from cbam import CBAMBlock
import settings


class NoCBAMBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()

    def forward(self, x):
        return x


CBAM = CBAMBlock if settings.use_se else NoCBAMBlock


class ConvDirec(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad = int(dilation * (kernel - 1) / 2)
        self.conv = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad, dilation=dilation)
        # print("oup_dim:",oup_dim)
        self.cbam = CBAMBlock(oup_dim, 6)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        x = self.conv(x)
        x = self.relu(self.cbam(x))
        return x, None


class ConvUnit(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv1 = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        # self.cbam = CBAMBlock("FC", 5, channels=oup_dim, ratio=4)
        self.cbam=CBAMBlock("Conv", 5, channels = oup_dim, gamma = 2, b = 1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.relu(self.cbam(x))
        return x


class LDVS(nn.Module):
    def __init__(self):
        super().__init__()
        channel = settings.channel
        self.enterBlock = nn.Sequential(nn.Conv2d(3, channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.rnns = nn.ModuleList(
            [ConvUnit(channel, channel, 3, 2 ** i) for
             i in range(settings.depth - 3)]
        )
        self.exitBlock = nn.Sequential(nn.Conv2d(channel, 3, 3, 1, 1), nn.LeakyReLU(0.2))
    def forward(self, x):
        ori = x
        image_feature = self.enterBlock(x)
        for rnn in self.rnns:
            image_feature = rnn(image_feature)
        rain = self.exitBlock(image_feature)
        derain = ori - rain
        return derain


if __name__ == '__main__':
    from torchsummary import summary
    ts = torch.Tensor(16, 3, 64, 64)
    vr = Variable(ts)
    net = LDVS()
    # print(net)
    oups = net(vr)
    print(oups.size())
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(num_params )
    summary(net.cuda(),(3, 64, 64))

