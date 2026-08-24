import pytest
import torch

import flag_gems

from . import base, consts


def npu_log_normal(x, mean=1.0, std=2.0):
    out = torch.empty_like(x)
    out.normal_(mean=0.0, std=1.0)
    out.mul_(std)
    out.add_(mean)
    out.exp_()
    return out


@pytest.mark.log_normal
def test_log_normal():
    torch_op = npu_log_normal if flag_gems.device == "npu" else torch.Tensor.log_normal_
    bench = base.GenericBenchmark(
        op_name="log_normal",
        torch_op=torch_op,
        gems_op=flag_gems.log_normal,
        input_fn=base.unary_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
