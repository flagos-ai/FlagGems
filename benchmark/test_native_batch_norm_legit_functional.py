import pytest
import torch

from . import base, consts


class NormBenchmark(base.GenericBenchmark):
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


@pytest.mark.native_batch_norm_legit_functional
def test_native_batch_norm_legit_functional():
    def native_batch_norm_legit_functional_input_fn(shape, dtype, device):
        C = shape[1]
        inp = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(C, dtype=dtype, device=device)
        bias = torch.randn(C, dtype=dtype, device=device)
        running_mean = torch.zeros(C, dtype=dtype, device=device)
        running_var = torch.ones(C, dtype=dtype, device=device)
        yield inp, weight, bias, running_mean, running_var, True, 0.1, 1e-5

    bench = NormBenchmark(
        input_fn=native_batch_norm_legit_functional_input_fn,
        op_name="native_batch_norm_legit_functional",
        torch_op=torch.ops.aten._native_batch_norm_legit_functional.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    from flag_gems.ops._native_batch_norm_legit_functional import (
        _native_batch_norm_legit_functional as gems_bn,
    )

    bench.set_gems(gems_bn)
    bench.run()
