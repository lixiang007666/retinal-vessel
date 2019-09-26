from utils.layer_utils import *


class Unet(nn.Module):

    def __init__(self, in_channels, num_class=2):
        super(Unet, self).__init__()

        self.num_class = num_class

        self.b1 = nn.Sequential(
            ConvBNAct(in_channels, 32, kernel_size=(3, 3), padding=(1, 1)),
            ConvBNAct(32, 32, kernel_size=(3, 3), padding=(1, 1)),
        )
        self.p1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.b2 = nn.Sequential(
            ConvBNAct(32, 20, kernel_size=(3, 3), padding=(1, 1)),
            ConvBNAct(20, 20, kernel_size=(3, 3), padding=(1, 1)),
        )
        self.p2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.b3 = nn.Sequential(
            ConvBNAct(20, 12, kernel_size=(3, 3), padding=(1, 1)),
            ConvBNAct(12, 12, kernel_size=(3, 3), padding=(1, 1)),
        )
        self.p3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.b4 = nn.Sequential(
            ConvBNAct(12, 12, kernel_size=(3, 3), padding=(1, 1)),
            ConvBNAct(12, 12, kernel_size=(3, 3), padding=(1, 1)),
        )
        self.p4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.b5 = nn.Sequential(
            ConvBNAct(12, 12, kernel_size=(3, 3), padding=(1, 1)),
            # ConvBNAct(12, 12, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.d4 = DeconvConcatBNAct(12, 12, 12)
        self.d4_1 = nn.Sequential(
            ConvBNAct(24, 12, kernel_size=(3, 3), padding=(1, 1)),
            ConvBNAct(12, 12, kernel_size=(3, 3), padding=(1, 1))
        )

        self.d3 = DeconvConcatBNAct(12, 12, 12)
        self.d3_1 = nn.Sequential(
            ConvBNAct(24, 12, kernel_size=(3, 3), padding=(1, 1)),
            ConvBNAct(12, 12, kernel_size=(3, 3), padding=(1, 1))
        )

        self.d2 = DeconvConcatBNAct(12, 20, 20)
        self.d2_1 = nn.Sequential(
            ConvBNAct(40, 20, kernel_size=(3, 3), padding=(1, 1)),
            ConvBNAct(20, 20, kernel_size=(3, 3), padding=(1, 1))
        )

        self.d1 = DeconvConcatBNAct(20, 32, 20)
        self.d1_1 = nn.Sequential(
            ConvBNAct(52, 52, kernel_size=(3, 3), padding=(1, 1)),
            ConvBNAct(52, 52, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.d0 = ConvBNAct(52, 32, kernel_size=(3, 3), padding=(1, 1))
        self.out = nn.Conv2d(32, self.num_class,  kernel_size=(1, 1), stride=1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        out1 = self.b1(x)
        down1 = self.p1(out1)
        out2 = self.b2(down1)
        down2 = self.p2(out2)
        out3 = self.b3(down2)
        down3 = self.p3(out3)
        out4 = self.b4(down3)
        down4 = self.p4(out4)
        out5 = self.b5(down4)

        out6 = self.d4_1(self.d4(out5, out4))
        out7 = self.d3_1(self.d3(out6, out3))
        out8 = self.d2_1(self.d2(out7, out2))
        out9 = self.d1_1(self.d1(out8, out1))

        return self.softmax(self.out(self.d0(out9)))