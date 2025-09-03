import torch
import torch.nn as nn

class InvertedResidualBlock3D(nn.Module):
    """
    3D inverted residual block for MobileNetV2
    """
    def __init__(self, in_c, out_c, stride, expansion_factor=6, deconvolve=False):
        super(InvertedResidualBlock3D, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.use_skip = (stride == 1)
        ex_c = int(in_c * expansion_factor)
        if deconvolve:
            self.conv = nn.Sequential(
                # pointwise conv
                nn.Conv3d(in_c, ex_c, kernel_size=1, bias=False),
                nn.BatchNorm3d(ex_c),
                nn.ReLU6(inplace=True),
                # depthwise deconv
                nn.ConvTranspose3d(
                    ex_c, ex_c, kernel_size=4, stride=stride, padding=1,
                    groups=ex_c, bias=False
                ),
                nn.BatchNorm3d(ex_c),
                nn.ReLU6(inplace=True),
                # pointwise conv
                nn.Conv3d(ex_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_c),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_c, ex_c, kernel_size=1, bias=False),
                nn.BatchNorm3d(ex_c),
                nn.ReLU6(inplace=True),
                nn.Conv3d(
                    ex_c, ex_c, kernel_size=3, stride=stride, padding=1,
                    groups=ex_c, bias=False
                ),
                nn.BatchNorm3d(ex_c),
                nn.ReLU6(inplace=True),
                nn.Conv3d(ex_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_c),
            )
        self.skip_conv = nn.Conv3d(in_c, out_c, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(x)
        if self.use_skip:
            res = x
            if x.shape[1] != out.shape[1]:
                res = self.skip_conv(x)
            return res + out
        return out

class MobileUNet3D(nn.Module):
    """
    3D Mobile UNet with inverted residual and depthwise separable conv
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(MobileUNet3D, self).__init__()
        # encoding
        self.conv_init = self.depthwise_conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.irb1 = self._make_irb(32, 16, num_blocks=1, stride=1, t=1)
        self.irb2 = self._make_irb(16, 24, num_blocks=2, stride=2, t=6)
        self.irb3 = self._make_irb(24, 32, num_blocks=3, stride=2, t=6)
        self.irb4 = self._make_irb(32, 64, num_blocks=4, stride=2, t=6)
        self.irb5 = self._make_irb(64, 96, num_blocks=3, stride=1, t=6)
        self.irb6 = self._make_irb(96, 160, num_blocks=3, stride=2, t=6)
        self.irb7 = self._make_irb(160, 320, num_blocks=1, stride=1, t=6)
        self.conv_encode = nn.Conv3d(320, 1280, kernel_size=1)
        # decoding
        self.d_irb1 = self._make_irb(1280, 96, num_blocks=1, stride=2, t=6, deconv=True)
        self.d_irb2 = self._make_irb(96, 32, num_blocks=1, stride=2, t=6, deconv=True)
        self.d_irb3 = self._make_irb(32, 24, num_blocks=1, stride=2, t=6, deconv=True)
        self.d_irb4 = self._make_irb(24, 16, num_blocks=1, stride=2, t=6, deconv=True)
        self.deconv = nn.ConvTranspose3d(16, 16, kernel_size=4, stride=2, padding=1, groups=16, bias=False)
        self.conv_out = nn.Conv3d(16, out_channels, kernel_size=1)

    def depthwise_conv3d(self, in_c, out_c, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv3d(in_c, in_c, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=in_c),
            nn.BatchNorm3d(in_c),
            nn.ReLU6(inplace=True),
            nn.Conv3d(in_c, out_c, kernel_size=1),
        )

    def _make_irb(self, in_c, out_c, num_blocks, stride, t, deconv=False):
        layers = []
        layers.append(InvertedResidualBlock3D(in_c, out_c, stride, expansion_factor=t, deconvolve=deconv))
        for _ in range(1, num_blocks):
            layers.append(InvertedResidualBlock3D(out_c, out_c, 1, expansion_factor=t, deconvolve=deconv))
        return nn.Sequential(*layers)

    def forward(self, x):
        # encoder
        x1 = self.conv_init(x)
        x2 = self.irb1(x1)
        x3 = self.irb2(x2)
        x4 = self.irb3(x3)
        x5 = self.irb4(x4)
        x6 = self.irb5(x5)
        x7 = self.irb6(x6)
        x8 = self.irb7(x7)
        x9 = self.conv_encode(x8)
        # decoder with skip
        d1 = self.d_irb1(x9) + x6
        d2 = self.d_irb2(d1) + x4
        d3 = self.d_irb3(d2) + x3
        d4 = self.d_irb4(d3) + x2
        d5 = self.deconv(d4)
        out = self.conv_out(d5)
        return out


from torchsummary import summary
from thop import profile
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Img_size=128
    in_channels=1
    data = torch.randn((1, in_channels, Img_size, Img_size, Img_size), device=device)
    model = MobileUNet3D(in_channels=1, out_channels=3).float()
    model = model.to(device)
    print(model(data).size())
    summary(model, input_size=(in_channels, Img_size, Img_size, Img_size))
    # for name, param in model.named_parameters():
    #     print(name, param.numel())
    flops, params = profile(model, inputs=(data,))
    print(f"FLOPs: {flops}, Parameters: {params}")