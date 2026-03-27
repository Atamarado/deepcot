import numpy as np
import ptflops
from timm.models.vision_transformer import Mlp as timm_Mlp

from torch import nn
from torchaudio.transforms import FrequencyMasking
import continual as co

from src.models.passt.tool_block import GAP
from src.models.passt.passt_sed import PaSST_SED, InterpolateModule
from src.models.transformer.transformerXL import RelPositionMultiheadAttention, TransformerXL, RelPositionalEncoding, \
    RelPositionDeepCoTMultiheadAttention
from src.models.transformer_decoder import TransformerXLDecoder
from src.preprocess.augmentMelSTFT import AugmentMelSTFT
from src.models.passt.passt import Attention, Block, PaSST, PatchEmbed, Mlp, DeepCoTAttention

def conv_flops_counter_hook(conv_module, input, output, extra_per_position_flops=0, transpose=False):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims, dtype=np.int64)) * \
        (in_channels * filters_per_channel + extra_per_position_flops)

    if transpose:
        input_dims = list(input.shape[2:])
        active_elements_count = batch_size * int(np.prod(input_dims, dtype=np.int64))
    else:
        output_dims = list(output.shape[2:])
        active_elements_count = batch_size * int(np.prod(output_dims, dtype=np.int64))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:
        bias_flops = batch_size * int(np.prod(list(output.shape[1:]), dtype=np.int64))

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)

def attention_counter_hook(module, input, output):
    input = input[0]
    _, n, _ = input.shape
    d = module.attn.head_dim

    if module.continual:
        macs = module.attn.num_heads * (3*n*d + 2*n) / 2
    else:
        macs = module.attn.num_heads * (2*n*n*d + n*n + n*d + n) / 2
    module.__flops__ += int(macs)
    pass

CUSTOM_FLOP_MODULES = {co.Conv2d: conv_flops_counter_hook, Block: attention_counter_hook, TransformerXL: attention_counter_hook}

# Adapted from https://github.com/LukasHedegaard/continual-transformers/blob/0b6092dea5e897ac290f5ad507a6fd5a55f24dd0/continual_transformers/ptflops.py#L9
def _register_ptflops():
    try:
        import ptflops

        if hasattr(ptflops, "pytorch_ops"):  # >= v0.6.8
            fc = ptflops.pytorch_ops
        else:  # < v0.6.7
            fc = ptflops.flops_counter

        for k, v in CUSTOM_FLOP_MODULES.items():
            fc.CUSTOM_MODULES_MAPPING[k] = v

    except ModuleNotFoundError:  # pragma: no cover
        raise ModuleNotFoundError
    except Exception as e:  # pragma: no cover
        raise Exception(f"Failed to add flops_counter_hook: {e}")


_register_ptflops()

def get_model_complexity_info(*args, **kwargs):
    return ptflops.get_model_complexity_info(*args, **kwargs,
                                             custom_modules_hooks=ptflops.pytorch_ops.CUSTOM_MODULES_MAPPING)