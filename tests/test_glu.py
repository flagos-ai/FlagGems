import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.glu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_glu(shape, dtype):
    if flag_gems.vendor_name == "tsingmicro":
        res_inp = torch.randn(shape, dtype=dtype, device="cpu")
    else:
        res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    for dim in range(len(shape)):
        if shape[dim] % 2 != 0:
            continue

        ref_out = torch.nn.functional.glu(ref_inp, dim=dim)
        with flag_gems.use_gems():
            if flag_gems.vendor_name == "tsingmicro":
                res_out = torch.nn.functional.glu(res_inp.to(device=flag_gems.device), dim=dim)
            else:
                res_out = torch.nn.functional.glu(res_inp, dim=dim)
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.glu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_glu_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    for dim in range(len(shape)):
        if shape[dim] == 0 or shape[dim] % 2 != 0:
            continue
        out_shape = list(shape)
        out_shape[dim] //= 2
        res_out = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)
        ref_out = utils.to_reference(res_out, True)

        ref_in_grad = torch.ops.aten.glu_backward(ref_out, ref_inp, dim=dim)
        with flag_gems.use_gems():
            res_in_grad = torch.ops.aten.glu_backward(res_out, res_inp, dim=dim)

        utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.iluvatar
@pytest.mark.parametrize(
    "shape",
    [
        (2, 128),
        (4, 256),
        (1, 1024),
        (8, 65),  # 非对齐
    ],
)
def test_glu_fused_iluvatar_triton(shape):
    import torch
    from accuracy_utils import check_close

    from flag_gems.ops.mm_glu_ops import triton_glu_fused
    from flag_gems.runtime import torch_device_fn

    with torch_device_fn.device("iluvatar"):
        x = torch.randn(*shape, dtype=torch.bfloat16)
        ref = torch.nn.functional.glu(x, dim=-1)
        out = triton_glu_fused(x)
        check_close(out, ref)
