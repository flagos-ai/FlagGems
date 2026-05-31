import pytest
import torch

import flag_gems

from . import base, consts


class NormBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            # 3D shapes: [batch_size, channels, hidden_size]
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            # 4D shapes: [batch_size, channels, H, W]
            (1, 8, 4, 4),
            (16, 8, 128, 128),
        ]


def layernorm_simple_input_fn(shape, dtype, device):
    # Simplified LayerNorm that accepts (input, normalized_shape, weight, bias, eps)
    inp = torch.randn(shape, dtype=dtype, device=device)
    layer_shape = (shape[-1],)
    weight = torch.randn(layer_shape, dtype=dtype, device=device)
    bias = torch.randn(layer_shape, dtype=dtype, device=device)
    yield inp, layer_shape, weight, bias


@pytest.mark.LayerNorm
def test_LayerNorm():
    bench = NormBenchmark(
        input_fn=layernorm_simple_input_fn,
        op_name="LayerNorm",
        torch_op=torch.nn.functional.layer_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.LayerNorm)
    bench.run()
