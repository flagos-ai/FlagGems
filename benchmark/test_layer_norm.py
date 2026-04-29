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
