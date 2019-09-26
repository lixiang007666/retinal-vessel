import torch
from torch import nn

class ConvBNAct(nn.Module):

    def __init__(self, in_channels, out_channels, *args, act='relu', **kwargs):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, *args, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leaky':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError('unknown activation name')

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DeconvConcatBNAct(nn.Module):

    def __init__(self, in_channels, sub_channels, out_channels, *args, act='relu', **kwargs):
        super(DeconvConcatBNAct, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels + sub_channels)

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leaky':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError('unknown activation name')

    def forward(self, x, sub):
        expand = self.deconv(x)
        o = torch.cat([expand, sub], dim=1)
        return self.act(self.bn(o))

# 定义权值初始化
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0, 0.01)
            m.bias.data.zero_()

