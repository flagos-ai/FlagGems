# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Generator

import pytest
import torch

from flag_gems.ops.quantized_max_pool2d import (
    _parse_pool_params,
    max_pool2d_output_size,
)

from . import base, consts

# quantized_max_pool2d consumes per-tensor quantized tensors (torch.quint8 /
# torch.qint8). The native quantized pooling kernel only runs on CPU in this
# build, so the input is kept on CPU; the FlagGems kernel moves the integer
# representation onto the accelerator and returns a CPU quantized tensor.
QUANT_DTYPES = [torch.quint8, torch.qint8]

# Spatial shapes exercised by the quantized pooling benchmark, spanning the
# typical ResNet stage outputs from the input image down to the final layer.
POOL_SHAPES = [
    (4, 3, 224, 224),  # Typical input image size
    (16, 64, 56, 56),  # Early ResNet layer output
    (32, 128, 28, 28),  # Mid ResNet layer output
    (64, 256, 14, 14),  # Later ResNet layer output
    (128, 512, 7, 7),  # Final ResNet layer output
]


def _pool_configs(shape, dtype, comprehensive):
    """Yield (kernel_size, stride, padding, dilation, ceil_mode) pooling configs.

    Mirrors the representative pooling configurations used by the forward
    benchmark: a default 3x3 stride-2 pool, plus extra kernel/stride/padding
    and dilation/ceil_mode variants under the comprehensive bench level.
    """
    yield 3, 2, 1, 1, False

    if comprehensive:
        if shape[-2] > 5 and shape[-1] > 5:
            yield (3, 5), (2, 1), (1, 2), 1, False
        yield 3, 1, 1, 2, False
        yield 3, 2, 1, 1, True


class QuantizedMaxPool2dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        comprehensive = base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
        for shape in POOL_SHAPES:
            yield from quantized_max_pool2d_input_fn(
                shape, dtype, self.device, comprehensive
            )


def quantized_max_pool2d_input_fn(shape, dtype, device, comprehensive=True):
    if dtype == torch.quint8:
        data = torch.rand(shape) * 255.0
        zero_point = 128
    else:
        data = torch.randint(-128, 128, shape).float()
        zero_point = 0
    scale = 0.1
    # Quantized tensors are kept on CPU because the native quantized pooling
    # kernel is CPU-only in this build; the FlagGems implementation handles the
    # accelerator round-trip internally.
    inp = torch.quantize_per_tensor(data, scale, zero_point, dtype)

    for kernel_size, stride, padding, dilation, ceil_mode in _pool_configs(
        shape, dtype, comprehensive
    ):
        yield inp, {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "ceil_mode": ceil_mode,
        }


class QuantizedMaxPool2dOutBenchmark(base.GenericBenchmark):
    """Benchmark for the aten::quantized_max_pool2d.out variant.

    The out variant writes the pooled integer representation into a caller
    pre-allocated quantized tensor; the benchmark pre-allocates that tensor
    with the correct output shape (derived from the pooling parameters) and
    the input's scale/zero_point, and passes it through the ``out`` kwarg.
    """

    def get_input_iter(self, dtype) -> Generator:
        comprehensive = base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
        for shape in POOL_SHAPES:
            yield from quantized_max_pool2d_out_input_fn(
                shape, dtype, self.device, comprehensive
            )


def quantized_max_pool2d_out_input_fn(shape, dtype, device, comprehensive=True):
    if dtype == torch.quint8:
        data = torch.rand(shape) * 255.0
        zero_point = 128
    else:
        data = torch.randint(-128, 128, shape).float()
        zero_point = 0
    scale = 0.1
    inp = torch.quantize_per_tensor(data, scale, zero_point, dtype)

    for kernel_size, stride, padding, dilation, ceil_mode in _pool_configs(
        shape, dtype, comprehensive
    ):
        (
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
        ) = _parse_pool_params(kernel_size, stride, padding, dilation)
        _, _, in_h, in_w = shape
        out_h = max_pool2d_output_size(
            in_h, kernel_h, stride_h, padding_h, dilation_h, ceil_mode
        )
        out_w = max_pool2d_output_size(
            in_w, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
        )
        # Pre-allocate the quantized ``out`` tensor with the correct output
        # shape and the input's scale/zero_point via the public API. The
        # FlagGems out kernel overwrites its integer storage.
        out_tensor = torch.quantize_per_tensor(
            torch.zeros((*shape[:2], out_h, out_w)), scale, zero_point, dtype
        )
        yield inp, {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "ceil_mode": ceil_mode,
            "out": out_tensor,
        }


@pytest.mark.quantized_max_pool2d
def test_quantized_max_pool2d():
    bench = QuantizedMaxPool2dBenchmark(
        op_name="quantized_max_pool2d",
        torch_op=torch.quantized_max_pool2d,
        dtypes=QUANT_DTYPES,
        input_fn=quantized_max_pool2d_input_fn,
    )
    bench.run()


@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out():
    # The out variant writes the pooled result into the caller-provided quantized
    # tensor. It dispatches through the aten .out overload, which takes the out
    # tensor via the ``out`` keyword; flag_gems registers it as
    # ``quantized_max_pool2d.out`` and the use_gems() context routes the dispatch
    # to the Triton kernel.
    bench = QuantizedMaxPool2dOutBenchmark(
        op_name="quantized_max_pool2d_out",
        torch_op=torch.ops.aten.quantized_max_pool2d.out,
        dtypes=QUANT_DTYPES,
        input_fn=quantized_max_pool2d_out_input_fn,
    )
    bench.run()
