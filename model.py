import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torchsummary import summary
import settings
from ptflops import get_model_complexity_info
from thop import profile



class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction = 16, spatial_kernel = 3):
        super().__init__()
        
        #channel attention 
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        #shared MLP
        self.mlp = nn.Sequential(
            #Conv2d比Linear方便操作
            #nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel,channel//reduction,1,bias = False),
            #inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            #nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel//reduction,channel,1,bias = False)
        )
        
        #spatial attention
        self.conv = nn.Conv2d(2,1,kernel_size = spatial_kernel, padding = spatial_kernel//2,bias=False)
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self,x):
        
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out+avg_out)
        x = channel_out * x
        
        max_out,_ = torch.max(x,dim = 1,keepdim=True)
        avg_out = torch.mean(x,dim = 1,keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out,avg_out],dim = 1)))
        x = spatial_out * x
        return x



CBAM = CBAMBlock 


class ConvDirec(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad = int(dilation * (kernel - 1) / 2)
        self.conv = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad, dilation=dilation)  
        self.cbam = CBAM(oup_dim, 6)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        x = self.conv(x)
        x = self.relu(self.cbam(x))
        return x, None





class ConvGRU(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_xz = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xr = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xn = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_hz = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hr = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hn = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

        self.cbam = CBAM(oup_dim, 6)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        if h is None:
            z = F.sigmoid(self.conv_xz(x))
            f = F.tanh(self.conv_xn(x))
            h = z * f
        else:
            z = F.sigmoid(self.conv_xz(x) + self.conv_hz(h))
            r = F.sigmoid(self.conv_xr(x) + self.conv_hr(h))
            n = F.tanh(self.conv_xn(x) + self.conv_hn(r * h))
            h = (1 - z) * h + z * n

        h = self.relu(self.cbam(h))
        return h, h




RecUnit = {
    'Conv': ConvDirec,  
    'GRU': ConvGRU, 
}[settings.uint]


class LDVS(nn.Module):
    def __init__(self):
        super().__init__()
        channel = settings.channel

        self.rnns = nn.ModuleList(
            [RecUnit(3, channel, 3, 1)] + 
            [RecUnit(channel, channel, 3, 2 ** i) for i in range(settings.depth - 3)]
        )

        self.dec = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            CBAM(channel, 6),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel, 3, 1),
        )

    def forward(self, x):
        ori = x
        old_states = [None for _ in range(len(self.rnns))]
        oups = []

        for i in range(settings.stage_num):
            states = []
            for rnn, state in zip(self.rnns, old_states):
                x, st = rnn(x, state)
                states.append(st)
            x = self.dec(x)
            
            if settings.frame == 'Add' and i > 0:
                x = x + Variable(oups[-1].data)

            oups.append(x)
            old_states = states.copy()
            x = ori - x

        return oups


if __name__ == '__main__':
    ts = torch.Tensor(16, 3, 64, 64)
    vr = Variable(ts)
    net = LDVS()
    print(net)
    oups = net(vr)
    for oup in oups:
        print(oup.size())
   
