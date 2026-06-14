import pytest
import torch

from . import base, consts

from flag_gems.ops.special_log1p import special_log1p


@pytest.mark.special_log1p
def test_special_log1p():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_log1p",
        torch_op=torch.special.log1p,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_log1p
def test_special_log1p_non_tensor():
    inp = 1.0
    ref_out = torch.special.log1p(torch.tensor(inp))
    res_out = special_log1p(inp)
    torch.testing.assert_close(ref_out, res_out)
