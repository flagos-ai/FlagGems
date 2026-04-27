from typing import Generator

import pytest
import torch

import flag_gems

from . import performance_utils as base
from . import utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    pass


class ScaledSoftmaxBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        # shape: [batch, heads, query_len, key_len]
        shapes_small = [
            (1, 4, 64, 64),
            (2, 8, 128, 128),
            (4, 8, 256, 256),
        ]
        shapes_medium = [
            (8, 12, 512, 512),
            (16, 16, 1024, 1024),
            (32, 16, 512, 512),
        ]
        shapes_large = [
            (1, 32, 2048, 2048),
            (2, 40, 4096, 4096),
            # (4, 32, 8192, 8192),  # too big shape, out of memory
        ]
        shapes_4d = shapes_small + shapes_medium + shapes_large
        for shape in shapes_4d:
            yield from self.input_fn(shape, dtype, self.device)


def scaled_softmax_forward_input_fn(shape, dtype, device):
    S = utils.generate_tensor_input(shape, dtype, device)
    scale_factor = 1 / S.shape[-1] ** 0.5
    yield S, scale_factor


@pytest.mark.scaled_softmax_forward
@pytest.mark.skipif(TE_AVAILABLE is False, reason="TransformerEngine is not available")
def test_scaled_softmax_forward():
    bench = ScaledSoftmaxBenchmark(
        input_fn=scaled_softmax_forward_input_fn,
        op_name="scaled_softmax_forward",
        torch_op=tex.scaled_softmax_forward,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(flag_gems.scaled_softmax_forward)
    bench.run()


def scaled_softmax_backward_input_fn(shape, dtype, device):
    S = utils.generate_tensor_input(shape, dtype, device)
    scale_factor = 1 / S.shape[-1] ** 0.5
    P = torch.softmax(S / scale_factor, dim=-1)
    dP = utils.generate_tensor_input(shape, dtype, device)
    yield P, dP, scale_factor


@pytest.mark.scaled_softmax_backward
@pytest.mark.skipif(TE_AVAILABLE is False, reason="TransformerEngine is not available")
def test_perf_scaled_softmax_backward():
    bench = ScaledSoftmaxBenchmark(
        input_fn=scaled_softmax_backward_input_fn,
        op_name="scaled_softmax_backward",
        torch_op=tex.scaled_softmax_backward,
        dtypes=[torch.float16, torch.bfloat16],
    )

    bench.set_gems(flag_gems.scaled_softmax_backward)

    bench.run()
