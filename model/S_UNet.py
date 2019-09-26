from model.UNet import *
from utils.layer_utils import *

class Mi_UNet(nn.Module):
    def __init__(self,in_channels, num_class=2):
        super(Mi_UNet, self).__init__()
        self.num_class = num_class
        self.unet0 = Unet(in_channels,self.num_class)
        self.unet1 = Unet(in_channels+1,self.num_class)
        self.unet2 = Unet(in_channels+2,self.num_class)

    def forward(self, inputs, **kwargs):
        o = self.unet0(inputs)

        nin0 = torch.unsqueeze(o[:, -1, :, :], 1)
        nin = torch.cat([nin0, inputs], 1)
        o1 = self.unet1(nin)

        nin1 = torch.unsqueeze(o1[:, -1, :, :], 1)
        nin = torch.cat([nin0, nin1, inputs], 1)
        o2 = self.unet2(nin)

        return o2