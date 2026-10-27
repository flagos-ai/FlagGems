from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts


def _as_tuple(value, ndim):
    """Broadcast a scalar spatial parameter to one entry per dimension."""
    if isinstance(value, (tuple, list)):
        assert len(value) == ndim, f"expected {ndim} values, got {value}"
        return tuple(value)
    return (value,) * ndim


def _out_input_fn(case, dtype, device) -> Generator:
    """Same case, but with the trailing ``out`` kwarg the out variant takes."""
    for args in _input_fn(case, dtype, device):
        yield (*args, {"out": torch.empty(0, dtype=dtype, device=device)})


def _out_baseline(inp, weight, z, alpha, bias, stride, padding, dilation, groups, out):
    return torch.ops.aten.cudnn_convolution_add_relu.out(
        inp, weight, z, alpha, bias, stride, padding, dilation, groups, out=out
    )


def _fused_baseline(inp, weight, z, alpha, bias, stride, padding, dilation, groups):
    """Baseline: the aten fused operator this operator reimplements.

    It is the exact op under test rather than a composition, so the speedup
    reported is against the cuDNN fused kernel. It has no CPU or non-cuDNN
    path, so the benchmark is GPU-only by nature.
    """
    return torch.ops.aten.cudnn_convolution_add_relu(
        inp, weight, z, alpha, bias, stride, padding, dilation, groups
    )


BENCH_DTYPES = [
    torch.bfloat16,
    torch.float16,
    torch.float32,
]


# (input shape, out_c, kernel, stride, padding, dilation, groups). The core set
# is the conv2d core set, so the fused operator is measured on the shapes the
# underlying convolution is already benchmarked on and the epilogue's cost is
# directly attributable; 1x1 pointwise, depthwise and small-C_in are added
# because they walk different kernels inside cudnn_convolution.
#
# There is no 1D case: the baseline is the aten fused operator itself (see
# _fused_baseline), and on CUDA it raises CUDNN_STATUS_BAD_PARAM for rank-3
# input, so a 1D case would fail before the comparison could be timed.
CORE_CASES = [
    ((16, 64, 56, 56), 128, 1, 1, 0, 1, 1),
    ((8, 256, 64, 64), 256, 3, 1, 1, 1, 1),
    ((16, 32, 32, 32), 32, 3, 1, 2, 2, 1),
    ((8, 3, 224, 224), 16, 3, 1, 1, 1, 1),
    ((16, 32, 56, 56), 32, 3, 1, 1, 1, 32),
]


MORE_CASES = [
    ((32, 64, 128, 128), 32, 3, 1, 2, 1, 1),
    ((16, 32, 32, 32), 32, (3, 5), (2, 1), (1, 2), (1, 2), 1),
    ((32, 64, 210, 210), 16, 5, 2, 1, 1, 1),
    ((16, 32, 24, 24), 24, 3, 2, 2, 1, 2),
    ((2, 16, 16, 16, 16), 16, 3, 1, 1, 1, 1),
]


class CudnnConvAddReluBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        cases = list(CORE_CASES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            cases += MORE_CASES
        self.shapes = list(dict.fromkeys(cases))

    def get_input_iter(self, dtype) -> Generator:
        for case in self.shapes:
            yield from self.input_fn(case, dtype, self.device)


def _input_fn(case, dtype, device) -> Generator:
    """Build the positional argument tuple for one case.

    Both the baseline and flag_gems.cudnn_convolution_add_relu take
    (input, weight, z, alpha, bias, stride, padding, dilation, groups), so the
    same argument list drives both.
    """
    in_shape, out_c, kernel, stride, padding, dilation, groups = case
    ndim = len(in_shape) - 2
    weight_shape = (out_c, in_shape[1] // groups, *_as_tuple(kernel, ndim))
    stride = list(_as_tuple(stride, ndim))
    padding = list(_as_tuple(padding, ndim))
    dilation = list(_as_tuple(dilation, ndim))

    inp = torch.randn(in_shape, dtype=dtype, device=device)
    weight = torch.randn(weight_shape, dtype=dtype, device=device)
    # z and bias are sized to the convolution's output extent. Deriving it with
    # the functional convolution keeps this from duplicating the size formula
    # and works for every rank in the case list.
    conv_fn = {
        1: torch.nn.functional.conv1d,
        2: torch.nn.functional.conv2d,
        3: torch.nn.functional.conv3d,
    }[ndim]
    out_shape = conv_fn(
        inp,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    ).shape
    z = torch.randn(out_shape, dtype=dtype, device=device)
    bias = torch.randn(out_c, dtype=dtype, device=device)

    yield (inp, weight, z, 1.0, bias, stride, padding, dilation, groups)


@pytest.mark.cudnn_convolution_add_relu
def test_cudnn_convolution_add_relu():
    torch.backends.cudnn.allow_tf32 = False
    bench = CudnnConvAddReluBenchmark(
        input_fn=_input_fn,
        op_name="cudnn_convolution_add_relu",
        torch_op=_fused_baseline,
        dtypes=BENCH_DTYPES,
        gems_op=flag_gems.cudnn_convolution_add_relu,
    )
    bench.run()


@pytest.mark.cudnn_convolution_add_relu_out
def test_cudnn_convolution_add_relu_out():
    torch.backends.cudnn.allow_tf32 = False
    bench = CudnnConvAddReluBenchmark(
        input_fn=_out_input_fn,
        op_name="cudnn_convolution_add_relu_out",
        torch_op=_out_baseline,
        dtypes=BENCH_DTYPES,
        gems_op=flag_gems.cudnn_convolution_add_relu_out,
    )
    bench.run()
