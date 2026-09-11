import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# FP16/BF16 only: CUDA quantized matmul requires half-precision activation (no float32 support)
MIXED_DTYPES_LINEAR_DTYPES = [d for d in utils.FLOAT_DTYPES if d != torch.float32]


# Representative shapes covering int8/int4, with/without bias, various activations.
# (M, K, N, mode, has_bias, activation) where mode in ('int8', 'int4')
MIXED_DTYPES_LINEAR_SHAPES = [
    (16, 128, 64, "int8", True, "none"),
    (32, 256, 128, "int8", False, "relu"),
    (16, 128, 64, "int8", True, "silu"),
    (16, 128, 128, "int4", True, "none"),
    (32, 256, 256, "int4", True, "relu"),
    (64, 256, 128, "int8", True, "silu"),
]


@pytest.mark.mixed_dtypes_linear
@pytest.mark.parametrize(
    "M, K, N, mode, has_bias, activation", MIXED_DTYPES_LINEAR_SHAPES
)
@pytest.mark.parametrize("dtype", MIXED_DTYPES_LINEAR_DTYPES)
def test_mixed_dtypes_linear(M, K, N, mode, has_bias, activation, dtype):
    input_tensor = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    # int8 mode: weight [K, N] uint8; int4 mode: weight [K, N//2] uint8 (packed)
    ncols = N if mode == "int8" else N // 2
    weight = torch.randint(
        0, 256, (K, ncols), dtype=torch.uint8, device=flag_gems.device
    )
    scale = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device) if has_bias else None

    ref_input = utils.to_reference(input_tensor, False)
    ref_weight = utils.to_reference(weight, False)
    ref_scale = utils.to_reference(scale, False)
    ref_bias = utils.to_reference(bias, False) if bias is not None else None

    res_out = flag_gems.mixed_dtypes_linear(
        input_tensor, weight, scale, bias=bias, activation=activation
    )

    ref_out = _ref_mixed_dtypes_linear(
        ref_input, ref_weight, ref_scale, mode, ref_bias, activation
    )
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


def _ref_mixed_dtypes_linear(input, weight, scale, mode, bias, activation):
    """Reference implementation matching kernel's order of operations.

    The kernel uses different scale placement for fp16 vs bf16:
    - fp16: epilogue scale (matmul in fp32, then scale)
    - bf16: per-tile in-loop scale (scale each weight tile before matmul)
    """
    K, ncols = weight.shape
    N = scale.shape[0]
    if mode == "int8":
        # FasterTransformer biased int8: (uint8 - 128)
        w = weight.to(torch.int32) - 128
    else:  # int4
        # FasterTransformer biased int4: (uint4 - 8)
        lo = (weight.to(torch.int32) & 0xF) - 8
        hi = ((weight.to(torch.int32) >> 4) & 0xF) - 8
        w = torch.stack([lo, hi], dim=-1).reshape(K, N)

    if input.dtype == torch.float16:
        # fp16 path: epilogue scale
        wf = w.to(input.dtype)
        acc = torch.matmul(input.to(torch.float32), wf.to(torch.float32))
        acc = acc * scale.to(torch.float32)[None, :]
    else:
        # bf16 path: per-tile in-loop scale (apply scale before matmul)
        wdq = w.to(torch.float32) * scale.to(torch.float32)[None, :]
        wdq = wdq.to(input.dtype)
        acc = torch.matmul(input.to(torch.float32), wdq.to(torch.float32))

    if bias is not None:
        acc = acc + bias.to(torch.float32)[None, :]
    if activation == "relu":
        acc = torch.relu(acc)
    elif activation == "silu":
        acc = acc * torch.sigmoid(acc)
    return acc.to(input.dtype)
