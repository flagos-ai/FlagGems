import pytest
import torch

from . import base, consts, utils


def nll_loss_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    target_shape = list(shape)
    del target_shape[1]
    target = torch.randint(0, shape[-1], target_shape, device=device)
    yield inp, target

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        weight = torch.randn(shape[1], dtype=cur_dtype, device=device)
        yield inp, target, {"weight": weight, "ignore_index": 1, "reduction": "none"}


@pytest.mark.nll_loss_forward
def test_nll_loss_forward():
    bench = base.GenericBenchmark2DOnly(
        op_name="nll_loss",
        input_fn=nll_loss_input_fn,
        torch_op=torch.nn.functional.nll_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
