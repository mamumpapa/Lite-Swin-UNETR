import math
import re
from collections import namedtuple
from .layers import *


GlobalParams = namedtuple('GlobalParams', [
    'batch_norm_momentum',
    'batch_norm_epsilon',
    'dropout_rate',
    'num_classes',
    'width_coefficient',
    'depth_coefficient',
    'depth_divisor',
    'min_depth',
    'drop_connect_rate'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = namedtuple('BlockArgs', [
    'kernel_size',
    'num_repeat',
    'input_filters',
    'output_filters',
    'expand_ratio',
    'id_skip',
    'strides',  # for 3D: [stride_d, stride_h, stride_w]
    'se_ratio'
])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

IMAGENET_WEIGHTS = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
}


def round_filters(filters, global_params):
    """Round number of filters"""
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth,
                      int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of repeats"""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def get_efficientnet_params(model_name, override_params=None):
    """Get efficientnet params based on model name"""
    params_dict = {
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    if model_name not in params_dict:
        raise KeyError(f'There is no model named {model_name}.')

    width_coefficient, depth_coefficient, _, dropout_rate = params_dict[model_name]
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=0.2,
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None
    )
    if override_params:
        global_params = global_params._replace(**override_params)

    decoder = BlockDecoder()
    return decoder.decode(blocks_args), global_params

# block strings used for EfficientNet
blocks_args = [
    'r1_k3_s11_e1_i32_o16_se0.25',
    'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25',
    'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25',
    'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
]

class BlockDecoder(object):
    """Block Decoder for 3D EfficientNet"""

    @staticmethod
    def _decode_block_string(block_string):
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        # For 3D: set depth stride = 1, height/width from string
        s_h, s_w = int(options['s'][0]), int(options['s'][1])
        strides = [1, s_h, s_w]

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            strides=strides,
            se_ratio=float(options['se']) if 'se' in options else None
        )

    @staticmethod
    def _encode_block_string(block):
        args = [
            f'r{block.num_repeat}',
            f'k{block.kernel_size}',
            # encode only spatial strides
            f's{block.strides[1]}{block.strides[2]}',
            f'e{block.expand_ratio}',
            f'i{block.input_filters}',
            f'o{block.output_filters}'
        ]
        if 0 < block.se_ratio <= 1:
            args.append(f'se{block.se_ratio}')
        if not block.id_skip:
            args.append('noskip')
        return '_'.join(args)

    def decode(self, string_list):
        assert isinstance(string_list, list)
        blocks = []
        for block_string in string_list:
            blocks.append(self._decode_block_string(block_string))
        return blocks

    def encode(self, blocks_args_list):
        block_strings = []
        for block in blocks_args_list:
            block_strings.append(self._encode_block_string(block))
        return block_strings
