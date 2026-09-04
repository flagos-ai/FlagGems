import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# conv_tbc shapes: (T, B, Cin, Cout, kW, pad) covering pad=0/1/2, single/multi
# batch, and kernel widths 1/3/4/5 to exercise the padding-shift and matmul paths.
CONV_TBC_SHAPES = [
    (5, 2, 3, 4, 3, 0),
    (5, 2, 3, 4, 3, 1),
    (8, 1, 16, 16, 3, 2),
    (16, 4, 32, 64, 5, 2),
    (32, 2, 64, 32, 1, 0),
    (7, 3, 8, 8, 4, 1),
]


@pytest.mark.conv_tbc
@pytest.mark.parametrize("shape", CONV_TBC_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conv_tbc(shape, dtype):
    T, B, Cin, Cout, kW, pad = shape
    inp = torch.randn((T, B, Cin), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((kW, Cin, Cout), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((Cout,), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.conv_tbc(ref_inp, ref_weight, ref_bias, pad)
    res_out = flag_gems.conv_tbc(inp, weight, bias, pad)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=Cin * kW)


@pytest.mark.conv_tbc_out
@pytest.mark.parametrize("shape", CONV_TBC_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conv_tbc_out(shape, dtype):
    T, B, Cin, Cout, kW, pad = shape
    inp = torch.randn((T, B, Cin), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((kW, Cin, Cout), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((Cout,), dtype=dtype, device=flag_gems.device)
    Tout = T + 2 * pad - kW + 1

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.ops.aten.conv_tbc.out(ref_inp, ref_weight, ref_bias, pad)
    res_out = torch.empty((Tout, B, Cout), dtype=dtype, device=flag_gems.device)
    flag_gems.conv_tbc_out(inp, weight, bias, pad, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=Cin * kW)
