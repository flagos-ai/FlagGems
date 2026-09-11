import pytest
import torch

import flag_gems
from flag_gems.ops.special_log1p import special_log1p

from . import base, consts


@pytest.mark.special_log1p
def test_special_log1p():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_log1p",
        torch_op=torch.ops.aten.special_log1p,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_log1p_out
def test_special_log1p_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_log1p_out",
        torch_op=torch.ops.aten.special_log1p.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_log1p
@pytest.mark.parametrize("inp", [1.0, 5, -0.5])
def test_special_log1p_non_tensor(inp):
    ref_out = torch.special.log1p(torch.tensor(inp, dtype=torch.float32))
    with flag_gems.use_gems():
        res_out = special_log1p(inp)
    atol = 1e-3 if inp == -0.5 else 1e-4
    assert torch.allclose(ref_out, res_out.to(torch.float32), atol=atol)
