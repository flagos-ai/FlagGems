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


def _case_fn(shape, dtype):
    del dtype
    channels = shape[1]
    yield base.BenchmarkCasePlan(
        shape={
            "input": shape,
            "weight": (channels,),
            "bias": (channels,),
            "running_mean": (channels,),
            "running_var": (channels,),
        },
        params={"training": True, "momentum": 0.1, "eps": 1e-5, "cudnn_enabled": True},
        builder_args=(shape,),
    )


def _build_inputs(plan, dtype, device):
    (shape,) = plan.builder_args
    channels = shape[1]
    return (
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn(channels, dtype=dtype, device=device),
        torch.randn(channels, dtype=dtype, device=device),
        torch.zeros(channels, dtype=dtype, device=device),
        torch.ones(channels, dtype=dtype, device=device),
        plan.params["training"],
        plan.params["momentum"],
        plan.params["eps"],
        plan.params["cudnn_enabled"],
    )


@pytest.mark.batch_norm_impl_index
def test__batch_norm_impl_index():
    bench = NormBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        op_name="_batch_norm_impl_index",
        torch_op=torch._batch_norm_impl_index,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
