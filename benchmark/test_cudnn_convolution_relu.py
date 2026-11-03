from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


def out_baseline(input, weight, bias, stride, padding, dilation, groups, out=None):
    """Baseline for the .out overload: the input fn preallocates the out tensor
    the way a caller would, and this runs the aten overload into it."""
    return torch.ops.aten.cudnn_convolution_relu.out(
        input, weight, bias, stride, padding, dilation, groups, out=out
    )


def cudnn_convolution_relu_no_bias_input_fn(case, dtype, device):
    in_shape, out_c, groups, stride, padding, dilation = case
    ndim = len(in_shape) - 2
    k = 3
    weight_shape = (out_c, in_shape[1] // groups, *([k] * ndim))
    inp = utils.generate_tensor_input(in_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)
    yield (
        inp,
        weight,
        None,
        _as_list(stride, ndim),
        _as_list(padding, ndim),
        _as_list(dilation, ndim),
        groups,
    )


def _as_list(value, ndim):
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value] * ndim


def cudnn_convolution_relu_out_input_fn(case, dtype, device):
    in_shape, out_c, k, padding = case
    ndim = len(in_shape) - 2
    weight_shape = (out_c, in_shape[1], *([k] * ndim))
    inp = utils.generate_tensor_input(in_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)
    bias = torch.randn(out_c, dtype=dtype, device=device)
    out = torch.empty(
        _conv_output_shape(in_shape, out_c, k, padding, 1, ndim),
        dtype=dtype,
        device=device,
    )
    # The last dict item becomes kwargs (see unpack_to_args_kwargs); the rest
    # are positional, in the .out overload's argument order minus out.
    yield (
        inp,
        weight,
        bias,
        _as_list(1, ndim),
        _as_list(padding, ndim),
        _as_list(1, ndim),
        1,
        {"out": out},
    )


def _conv_output_shape(in_shape, out_c, k, padding, dilation, ndim):
    shape = [in_shape[0], out_c]
    for i in range(ndim):
        shape.append((in_shape[2 + i] + 2 * padding - dilation * (k - 1) - 1) // 1 + 1)
    return shape


# The fused aten op only has cuDNN engines for these three float types
# (fp64/int8 fail with "unable to find an engine"), matching consts.FLOAT_DTYPES.
BENCH_DTYPES = consts.FLOAT_DTYPES


# (input, weight, groups, stride, padding, dilation) with 3x3 kernels. The
# first three are typical CNN blocks (ResNet-style stem, residual block,
# classifier head); the comprehensive-level extras exercise the strided /
# 1x1 / depthwise paths. Stride/padding/dilation are folded into each case
# instead of a full cross-product, as in benchmark/test_cudnn_convolution.py.
CORE_CASES = [
    ((8, 3, 224, 224), 64, 1, 1, 1, 1),
    ((16, 64, 56, 56), 64, 1, 1, 1, 1),
    ((16, 512, 7, 7), 512, 1, 1, 0, 1),
]


MORE_CASES = [
    ((32, 64, 128, 128), 32, 1, 2, 1, 1),  # strided downsample
    ((2, 4, 9, 9), 6, 1, 1, 0, 1),  # 1x1 kernel, C_in < 16
    ((1, 8, 6, 6), 8, 8, 1, 0, 1),  # depthwise
]


# Bias-free cases exercise the dedicated pointwise/depthwise/generic kernels
# underneath (the same dispatch cudnn_convolution uses). Same 3x3/1x1 mix.
NO_BIAS_CASES = [
    ((16, 64, 56, 56), 128, 1, 1, 0, 1),  # pointwise 1x1
    ((1, 8, 6, 6), 8, 8, 1, 0, 1),  # depthwise
    ((2, 8, 16, 16), 6, 1, 1, 1, 1),  # generic 3x3
]


# .out case: (input, out_c, kernel, padding). The out tensor is built from the
# padding-1 stride-1 output extents inside the input fn.
OUT_CASES = [
    ((16, 64, 56, 56), 64, 3, 1),
    ((8, 3, 224, 224), 64, 3, 1),
]


aten = torch.ops.aten.cudnn_convolution_relu


class CudnnConvolutionReluBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        cases = list(CORE_CASES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            cases += MORE_CASES
        self.shapes = list(dict.fromkeys(cases))

    def get_input_iter(self, dtype) -> Generator:
        for case in self.shapes:
            yield from cudnn_convolution_relu_input_fn(case, dtype, self.device)


class CudnnConvolutionReluNoBiasBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # One bias-free case at core level; the rest at comprehensive level.
        cases = [NO_BIAS_CASES[0]]
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            cases += NO_BIAS_CASES[1:]
        self.shapes = list(dict.fromkeys(cases))

    def get_input_iter(self, dtype) -> Generator:
        for case in self.shapes:
            yield from cudnn_convolution_relu_no_bias_input_fn(case, dtype, self.device)


class CudnnConvolutionReluOutBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(OUT_CASES)

    def get_input_iter(self, dtype) -> Generator:
        for case in self.shapes:
            yield from cudnn_convolution_relu_out_input_fn(case, dtype, self.device)


def cudnn_convolution_relu_input_fn(case, dtype, device):
    in_shape, out_c, groups, stride, padding, dilation = case
    ndim = len(in_shape) - 2
    k = 3
    weight_shape = (out_c, in_shape[1] // groups, *([k] * ndim))

    inp = utils.generate_tensor_input(in_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)
    bias = torch.randn(out_c, dtype=dtype, device=device)

    yield (
        inp,
        weight,
        bias,
        _as_list(stride, ndim),
        _as_list(padding, ndim),
        _as_list(dilation, ndim),
        groups,
    )


@pytest.mark.cudnn_convolution_relu
def test_cudnn_convolution_relu():
    torch.backends.cudnn.allow_tf32 = False
    bench = CudnnConvolutionReluBenchmark(
        input_fn=cudnn_convolution_relu_input_fn,
        op_name="cudnn_convolution_relu",
        torch_op=aten,
        dtypes=BENCH_DTYPES,
        gems_op=flag_gems.cudnn_convolution_relu,
    )
    bench.run()


@pytest.mark.cudnn_convolution_relu
def test_cudnn_convolution_relu_no_bias():
    torch.backends.cudnn.allow_tf32 = False
    bench = CudnnConvolutionReluNoBiasBenchmark(
        input_fn=cudnn_convolution_relu_no_bias_input_fn,
        op_name="cudnn_convolution_relu",
        torch_op=aten,
        dtypes=BENCH_DTYPES,
        gems_op=flag_gems.cudnn_convolution_relu,
    )
    bench.run()


@pytest.mark.cudnn_convolution_relu_out
def test_cudnn_convolution_relu_out():
    torch.backends.cudnn.allow_tf32 = False
    bench = CudnnConvolutionReluOutBenchmark(
        input_fn=cudnn_convolution_relu_out_input_fn,
        op_name="cudnn_convolution_relu_out",
        torch_op=out_baseline,
        dtypes=BENCH_DTYPES,
        gems_op=flag_gems.cudnn_convolution_relu_out,
    )
    bench.run()
