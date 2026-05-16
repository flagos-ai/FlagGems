import pytest
import torch

from . import base, consts, utils


def _dropout_backward_input_fn(shape, dtype, device):
    grad_output = utils.generate_tensor_input(shape, dtype, device)
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    scale = 1.0 / 0.5  # 1.0 / (1.0 - p), where p=0.5
    yield grad_output, mask, {"scale": scale}


@pytest.mark.dropout
def test_dropout():
    bench = base.UnaryPointwiseBenchmark(
        op_name="dropout", torch_op=torch.nn.Dropout(p=0.5), dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.dropout_backward
def test_dropout_backward():
    bench = base.GenericBenchmark(
        op_name="dropout_backward",
        input_fn=_dropout_backward_input_fn,
        torch_op=torch.ops.aten.native_dropout_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
