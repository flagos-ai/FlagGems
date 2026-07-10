import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


def generate_input(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.randn(shape, dtype=dtype, device=device).contiguous()


def filter_valid_shapes(shapes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    valid_shapes = []
    for shape in shapes:
        if not shape:
            continue
        if shape[-1] % 2 == 0:
            valid_shapes.append(shape)
    return valid_shapes


VALID_POINTWISE_SHAPES = filter_valid_shapes(utils.SWIGLU_SPECIAL_SHAPES)


@pytest.mark.swiglu
@pytest.mark.parametrize("shape", VALID_POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(not TE_AVAILABLE, reason="transformer engine is not available")
def test_swiglu(shape: tuple[int, ...], dtype: torch.dtype):
    torch.manual_seed(42)
    device = flag_gems.device

    input_tensor = generate_input(shape, dtype, device)

    # TransformerEngine's swiglu requires a 2-D input on some backends (e.g. musa),
    # while FlagGems supports arbitrary shapes by flattening to 2-D internally.
    # Reshape the reference input to 2-D and restore the original output shape so
    # the comparison stays valid across vendors.
    last_dim = shape[-1]
    H = last_dim // 2
    ref_input = input_tensor.view(-1, last_dim)
    te_forward = tex.swiglu(ref_input, quantizer=None).to(device)
    te_forward = te_forward.view(*shape[:-1], H)
    te_forward = utils.to_reference(te_forward)

    with flag_gems.use_gems():
        fg_forward = flag_gems.swiglu(input_tensor, quantizer=None)

    utils.gems_assert_close(fg_forward, te_forward, dtype)
