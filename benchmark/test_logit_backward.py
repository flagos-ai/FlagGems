import pytest
import torch

from flag_gems.ops.logit_backward import logit_backward_out

from . import base, consts


@pytest.mark.logit_backward
def test_logit_backward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logit_backward",
        torch_op=lambda a: torch.ops.aten.logit_backward(
            torch.ones_like(a), torch.rand_like(a), 1e-6
        ),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class LogitBackwardOutBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            self_tensor = torch.rand(shape, dtype=cur_dtype, device=self.device)
            grad = torch.randn_like(self_tensor)
            out = torch.empty_like(self_tensor)
            yield grad, self_tensor, out

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[1].shape)
        return torch.tensor(shape).prod().item()


@pytest.mark.logit_backward
def test_logit_backward_out():
    bench = LogitBackwardOutBenchmark(
        op_name="logit_backward",
        torch_op=lambda grad, self, out: out.copy_(
            torch.ops.aten.logit_backward(grad, self, 1e-6)
        ),
        gems_op=lambda grad, self, out: logit_backward_out(grad, self, 1e-6, out=out),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
