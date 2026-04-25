import pytest
import torch

import flag_gems

from . import attri_util as attr_utils
from . import performance_utils as utils


class NormBenchmark(utils.GenericBenchmark):
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
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((C,), dtype=dtype, device=device)
    bias = torch.randn((C,), dtype=dtype, device=device)
    running_mean = None
    running_var = None
    use_input_stats = True
    momentum = 0.1
    eps = 1e-5
    cudnn_enabled = True
    yield inp, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled
    if utils.Config.bench_level == utils.BenchLevel.COMPREHENSIVE:
        running_mean = torch.randn((C,), dtype=dtype, device=device)
        running_var = torch.randn((C,), dtype=dtype, device=device)
        yield inp, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled


@pytest.mark.instance_norm
def test_instance_norm(monkeypatch):
    if flag_gems.vendor_name == "kunlunxin" and utils.SkipVersion("torch", "<2.5"):
        pytest.skip(
            "BF16 is not supported in XPytorch 2.0. Please upgrade your PyTorch version >= 2.5"
        )

    if flag_gems.vendor_name == "mthreads":
        # Compatible with older versions of LLVM
        monkeypatch.env("DISABLE_LLVM_OPT", "1")

    bench = NormBenchmark(
        op_name="instance_norm",
        input_fn=input_fn,
        torch_op=torch.instance_norm,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.instance_norm)
    bench.run()
