from collections import OrderedDict
import torch
import torch.nn as nn
from .layers import double_conv3d, up_conv3d
from .efficientnet import EfficientNet3D

import torch.nn.functional as F

__all__ = [
    'EfficientUnet3D',
    'get_efficientunet3d_b0',
    'get_efficientunet3d_b1',
    'get_efficientunet3d_b2',
    'get_efficientunet3d_b3',
    'get_efficientunet3d_b4',
    'get_efficientunet3d_b5',
    'get_efficientunet3d_b6',
    'get_efficientunet3d_b7',
]


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):
        def hook(module, input, output):
            nonlocal count
            try:
                if hasattr(module, 'name') and module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-3:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output
                elif hasattr(module, 'name') and module.name == 'head_swish':
                    blocks.popitem()
                    blocks[module.name] = output
            except Exception:
                pass
        if not isinstance(module, (nn.Sequential, nn.ModuleList)) and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)
    model(x)
    for h in hooks:
        h.remove()
    return blocks


class EfficientUnet3D(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True, in_ch=2, dummy_size=None):
        super().__init__()
        self.encoder = encoder
        self.concat_input = concat_input
        if dummy_size is None:
            dummy_size = (1, in_ch, 128, 128, 128)
        # 1) 더미 입력으로 skip-connected feature들 추출
        device = next(encoder.parameters()).device
        B, C, D, H, W = dummy_size
        with torch.no_grad():
            x_dummy = torch.zeros(B, C, D, H, W, device=device)
            skips = list(get_blocks_to_be_concat(encoder, x_dummy).values())
        # skips: [feat1, feat2, feat3, feat4, maybe head_feat]
        # 채널들
        chs = [f.shape[1] for f in skips]  
        # 예) chs == [64, 128, 256, 512, 192]  (마지막은 head_swish 이후 채널)

        # 2) 레이어 초기화: chs[-1] → first upconv in-ch
        self.up_conv1 = up_conv3d(chs[-1], 512)
        self.double_conv1 = double_conv3d(chs[-2] + 512, 512)

        self.up_conv2 = up_conv3d(512, 256)
        self.double_conv2 = double_conv3d(chs[-3] + 256, 256)

        self.up_conv3 = up_conv3d(256, 128)
        self.double_conv3 = double_conv3d(chs[-4] + 128, 128)

        self.up_conv4 = up_conv3d(128, 64)
        self.double_conv4 = double_conv3d(chs[-5] + 64, 64)

        if self.concat_input:
            # (1) 업샘플된 x 채널 수 = 64 -> up_conv_input out_channels = 32
            self.up_conv_input = up_conv3d(64, 32)
            # (2) concat 후 채널 수 = 32 + 원본 입력 채널 C
            B, C, D, H, W = dummy_size if len(dummy_size)==5 else (1, *dummy_size)
            in_ch_input = 32 + C
            self.double_conv_input = double_conv3d(in_ch_input, 32)

        # 최종 conv
        final_in = 32 if self.concat_input else 64
        self.final_conv = nn.Conv3d(final_in, out_channels, kernel_size=1)

    @property
    def n_channels(self):
        # number of channels from encoder head
        return self.encoder.head_conv.out_channels

    @property
    def size(self):
        # dummy size placeholders; adjust as needed
        return [512 + self.n_channels,
                256 + 512,
                128 + 256,
                64 + 128,
                32 + 64 if self.concat_input else None,
                64]

    def forward(self, x):
        input_ = x
        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x = blocks.popitem()

        # UP + CONCAT 1
        x = self.up_conv1(x)
        skip = blocks.popitem()[1]
        # spatial alignment
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv1(x)

        # UP + CONCAT 2
        x = self.up_conv2(x)
        skip = blocks.popitem()[1]
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv2(x)

        # UP + CONCAT 3
        x = self.up_conv3(x)
        skip = blocks.popitem()[1]
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv3(x)

        # UP + CONCAT 4
        x = self.up_conv4(x)
        skip = blocks.popitem()[1]
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv4(x)

        # OPTIONAL INPUT CONCAT
        if self.concat_input:
            x = self.up_conv_input(x)
            if x.shape[2:] != input_.shape[2:]:
                x = F.interpolate(x, size=input_.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        x = self.final_conv(x)
        return x


def _get_unet3d_factory(block_name):
    def fn(in_ch=2, out_channels=2, concat_input=True, pretrained=True):
        encoder = EfficientNet3D.encoder(block_name,
                                         in_ch=in_ch,        # <-- 여기를 바꿔서 원하는 채널 수를 전달
                                         pretrained=pretrained)
        return EfficientUnet3D(encoder, out_channels=out_channels, concat_input=concat_input)
    return fn

get_efficientunet3d_b0 = _get_unet3d_factory('efficientnet-b0')
get_efficientunet3d_b1 = _get_unet3d_factory('efficientnet-b1')
get_efficientunet3d_b2 = _get_unet3d_factory('efficientnet-b2')
get_efficientunet3d_b3 = _get_unet3d_factory('efficientnet-b3')
get_efficientunet3d_b4 = _get_unet3d_factory('efficientnet-b4')
get_efficientunet3d_b5 = _get_unet3d_factory('efficientnet-b5')
get_efficientunet3d_b6 = _get_unet3d_factory('efficientnet-b6')
get_efficientunet3d_b7 = _get_unet3d_factory('efficientnet-b7')
