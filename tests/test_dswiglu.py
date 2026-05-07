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


@pytest.mark.dswiglu
@pytest.mark.skipif(not TE_AVAILABLE, reason="transformer engine is not available")
@pytest.mark.parametrize("shape", VALID_POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dswiglu(shape: tuple[int, ...], dtype: torch.dtype):
    torch.manual_seed(42)
    device = flag_gems.device

    input_tensor = generate_input(shape, dtype, device)

    grad_shape = list(shape)
    grad_shape[-1] = grad_shape[-1] // 2
    grad_output = generate_input(tuple(grad_shape), dtype, device)

    te_grad_input = tex.dswiglu(grad_output, input_tensor, quantizer=None).to(device)
    te_grad_input = utils.to_reference(te_grad_input)

    with flag_gems.use_gems():
        fg_grad_input = flag_gems.dswiglu(grad_output, input_tensor, quantizer=None)

    utils.gems_assert_close(fg_grad_input, te_grad_input, dtype)
