import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv3dSamePadding(nn.Conv3d):
    """3D Convolutions with same padding"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 name=None):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding=0,
                         dilation=dilation,
                         groups=groups,
                         bias=bias)
        # normalize stride and dilation to 3-tuple
        self.stride = self._triple(self.stride)
        self.dilation = self._triple(self.dilation)
        self.name = name

    def _triple(self, val):
        if isinstance(val, (list, tuple)) and len(val) == 3:
            return val
        return (val, val, val)

    def forward(self, x):
        # compute padding for D, H, W dims
        in_d, in_h, in_w = x.size()[2:]
        k_d, k_h, k_w = self.weight.size()[2:]
        s_d, s_h, s_w = self.stride
        d_d, d_h, d_w = self.dilation

        out_d = math.ceil(in_d / s_d)
        out_h = math.ceil(in_h / s_h)
        out_w = math.ceil(in_w / s_w)

        pad_d = max((out_d - 1) * s_d + (k_d - 1) * d_d + 1 - in_d, 0)
        pad_h = max((out_h - 1) * s_h + (k_h - 1) * d_h + 1 - in_h, 0)
        pad_w = max((out_w - 1) * s_w + (k_w - 1) * d_w + 1 - in_w, 0)

        # pad order: last dim first
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, [
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_d // 2, pad_d - pad_d // 2,
            ])
        return F.conv3d(x,
                         self.weight,
                         self.bias,
                         self.stride,
                         self.padding,
                         self.dilation,
                         self.groups)


class BatchNorm3d(nn.BatchNorm3d):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 name=None):
        super().__init__(num_features,
                         eps=eps,
                         momentum=momentum,
                         affine=affine,
                         track_running_stats=track_running_stats)
        self.name = name


def drop_connect(inputs, drop_connect_rate, training):
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - drop_connect_rate
    random_tensor = keep_prob
    # shape for 5D tensor
    random_tensor += torch.rand([batch_size, 1, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class MBConvBlock3D(nn.Module):
    """3D Mobile Inverted Residual Bottleneck Block"""

    def __init__(self, block_args, global_params, idx):
        super().__init__()
        block_name = f'blocks_{idx}_'
        self.block_args = block_args
        self.batch_norm_momentum = 1 - global_params.batch_norm_momentum
        self.batch_norm_epsilon = global_params.batch_norm_epsilon
        self.has_se = (self.block_args.se_ratio is not None) and (0 < self.block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        self.swish = Swish(block_name + '_swish')

        in_ch = self.block_args.input_filters
        out_ch = in_ch * self.block_args.expand_ratio
        # Expansion
        if self.block_args.expand_ratio != 1:
            self._expand_conv = Conv3dSamePadding(in_ch,
                                                  out_ch,
                                                  kernel_size=1,
                                                  bias=False,
                                                  name=block_name + 'expansion_conv')
            self._bn0 = BatchNorm3d(out_ch,
                                    momentum=self.batch_norm_momentum,
                                    eps=self.batch_norm_epsilon,
                                    name=block_name + 'expansion_batch_norm')

        # Depthwise conv
        k = self.block_args.kernel_size
        stride = self.block_args.strides
        self._depthwise_conv = Conv3dSamePadding(out_ch,
                                                 out_ch,
                                                 kernel_size=k,
                                                 stride=stride,
                                                 groups=out_ch,
                                                 bias=False,
                                                 name=block_name + 'depthwise_conv')
        self._bn1 = BatchNorm3d(out_ch,
                                momentum=self.batch_norm_momentum,
                                eps=self.batch_norm_epsilon,
                                name=block_name + 'depthwise_batch_norm')

        # SE
        if self.has_se:
            num_squeezed = max(1, int(self.block_args.input_filters * self.block_args.se_ratio))
            self._se_reduce = Conv3dSamePadding(out_ch,
                                                num_squeezed,
                                                kernel_size=1,
                                                name=block_name + 'se_reduce')
            self._se_expand = Conv3dSamePadding(num_squeezed,
                                                out_ch,
                                                kernel_size=1,
                                                name=block_name + 'se_expand')

        # Output
        final_ch = self.block_args.output_filters
        self._project_conv = Conv3dSamePadding(out_ch,
                                               final_ch,
                                               kernel_size=1,
                                               bias=False,
                                               name=block_name + 'output_conv')
        self._bn2 = BatchNorm3d(final_ch,
                                momentum=self.batch_norm_momentum,
                                eps=self.batch_norm_epsilon,
                                name=block_name + 'output_batch_norm')

    def forward(self, x, drop_connect_rate=None):
        identity = x
        if self.block_args.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self.swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self.swish(x)

        if self.has_se:
            x_squeezed = F.adaptive_avg_pool3d(x, 1)
            x_squeezed = self._se_expand(self.swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)

        in_f, out_f = self.block_args.input_filters, self.block_args.output_filters
        if self.id_skip and self.block_args.strides == 1 and in_f == out_f:
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate, self.training)
            x = x + identity
        return x


def double_conv3d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv3d(in_channels, out_channels):
    return nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


def custom_head3d(in_channels, out_channels):
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(in_channels, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(512, out_channels)
    )
