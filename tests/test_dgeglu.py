import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

vendor_name = flag_gems.vendor_name

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "dgeglu", None)
except ImportError:
    TE_OP = None


@pytest.mark.dgeglu
@pytest.mark.skipif(TE_OP is None, reason="'dgeglu' not found in TransformerEngine")
@pytest.mark.skipif(
    vendor_name == "kunlunxin",
    reason="Kunlunxin TE dgeglu delegates to gelu with different arg types (kFloat32 dtype vs Tensor); "
    "FlagGems kernel also fails to compile on XPU (xpu3-elfconv error)",
)
@pytest.mark.parametrize("shape", utils.GLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dgeglu(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    grad_output_shape = list(shape)
    grad_output_shape[-1] //= 2
    grad_output = torch.randn(
        tuple(grad_output_shape), dtype=dtype, device=flag_gems.device
    )
    ref_out = TE_OP(grad_output, input_tensor, None)
    ref_out = utils.to_reference(ref_out)
    with flag_gems.use_gems():
        res_out = flag_gems.dgeglu(grad_output, input_tensor)
    utils.gems_assert_close(res_out, ref_out, dtype)
