import pytest
import torch

import flag_gems

from . import base, consts


def npu_log_normal_(self, mean=1.0, std=2.0):
    # Ascend NPU does not support log_normal_ natively, emulate it with supported torch_npu ops.
    self.normal_(mean=0.0, std=1.0)
    self.mul_(std)
    self.add_(mean)
    self.exp_()
    return self


@pytest.mark.log_normal_
def test_log_normal_():
    torch_op = (
        npu_log_normal_ if flag_gems.device == "npu" else torch.Tensor.log_normal_
    )

    bench = base.GenericBenchmark(
        op_name="log_normal_",
        torch_op=torch_op,
        gems_op=flag_gems.log_normal_,
        input_fn=base.unary_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
