import pytest
import torch

import flag_gems

from . import base  # noqa: F401
from .base import GenericBenchmark
from .conftest import Config
from .consts import FLOAT_DTYPES, BenchLevel


class NormBenchmark(GenericBenchmark):
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


@pytest.mark.miopen_batch_norm
def test_miopen_batch_norm():
    def miopen_batchnorm_input_fn(shape, dtype, device):
        C = shape[1]
        inp = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn((C,), dtype=dtype, device=device)
        bias = torch.randn((C,), dtype=dtype, device=device)
        running_mean = torch.zeros((C,), dtype=dtype, device=device)
        running_var = torch.ones((C,), dtype=dtype, device=device)
        training = True
        momentum = 0.1
        eps = 1e-5
        yield inp, weight, bias, running_mean, running_var, training, momentum, eps

        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            running_mean = torch.randn((C,), dtype=dtype, device=device)
            running_var = torch.randn((C,), dtype=dtype, device=device)
            yield inp, weight, bias, running_mean, running_var, training, momentum, eps

    # Use native_batch_norm as the baseline since miopen_batch_norm is not available in PyTorch
    # Both operators have similar semantics
    bench = NormBenchmark(
        input_fn=miopen_batchnorm_input_fn,
        op_name="miopen_batch_norm",
        torch_op=torch.ops.aten.native_batch_norm,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.miopen_batch_norm)
    bench.run()
