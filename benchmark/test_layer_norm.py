import pytest
import torch

from . import base, consts


# TODO(Qiming): Extract this to a base class
class NormBenchmark(base.GenericBenchmark):
    # TODO: add new metric

    def set_more_shapes(self):
        return [
            # 3D shapes represented as [batch_size, channels, hidden_size]
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            # 4D shapes represented as [batch_size, channels, H, W]
            (1, 8, 4, 4),
            (16, 8, 128, 128),
        ]


def input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    layer_shape = shape[1:]
    weight = torch.randn(layer_shape, dtype=dtype, device=device)
    bias = torch.randn(layer_shape, dtype=dtype, device=device)
    yield inp, layer_shape, weight, bias


@pytest.mark.layer_norm
def test_layer_norm():
    bench = NormBenchmark(
        op_name="layer_norm",
        input_fn=input_fn,
        torch_op=torch.layer_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def input_fn_backward(shape, dtype, device):
    grad_out = torch.randn(shape, dtype=dtype, device=device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    normalized_shape = shape[1:]
    # mean and rstd are always float32 for native_layer_norm_backward
    mean = torch.randn(shape[0], dtype=torch.float32, device=device)
    rstd = torch.randn(shape[0], dtype=torch.float32, device=device)
    weight = torch.randn(normalized_shape, dtype=dtype, device=device)
    bias = torch.randn(normalized_shape, dtype=dtype, device=device)
    output_mask = [True, True, True]
    yield grad_out, inp, normalized_shape, mean, rstd, weight, bias, output_mask


@pytest.mark.layer_norm_backward
def test_layer_norm_backward():
    bench = NormBenchmark(
        op_name="layer_norm_backward",
        input_fn=input_fn_backward,
        torch_op=torch.ops.aten.native_layer_norm_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
