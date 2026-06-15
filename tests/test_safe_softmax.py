import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.safe_softmax
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("in_dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize(
    "dtype_arg_sel", ["none", "same", torch.float32, torch.float16, torch.bfloat16]
)
def test_safe_softmax(shape, in_dtype, dim, dtype_arg_sel):
    x = torch.randn(shape, dtype=in_dtype, device=flag_gems.device)
    if dtype_arg_sel == "none":
        dtype_arg = None
    elif dtype_arg_sel == "same":
        dtype_arg = in_dtype
    else:
        dtype_arg = dtype_arg_sel

    ref_x = utils.to_reference(x)
    if dtype_arg in (torch.float16, torch.bfloat16):
        ref_x = ref_x.float()
        ref_out = torch.ops.aten._safe_softmax(ref_x, dim, dtype=torch.float32)
        ref_out = ref_out.to(dtype_arg)
    else:
        ref_out = torch.ops.aten._safe_softmax(ref_x, dim, dtype=dtype_arg)

    with flag_gems.use_gems():
        act_out = torch.ops.aten._safe_softmax(x, dim, dtype=dtype_arg)
    expected_dtype = dtype_arg if dtype_arg is not None else in_dtype

    utils.gems_assert_close(act_out, ref_out, expected_dtype)
