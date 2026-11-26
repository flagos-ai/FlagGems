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

import flag_gems

from . import base, consts, utils

BENCH_DTYPES = [
    torch.bfloat16,
    torch.float16,
    torch.float32,
]

CONV1D_CORE_CASES = [
    ((4, 16, 256), 16, 1, 1, 0, 1, 1),
    ((16, 32, 512), 32, 3, 1, 2, 2, 1),
    ((32, 64, 512), 64, 3, 1, 1, 1, 1),
]

CONV1D_MORE_CASES = [
    ((16, 32, 1024), 32, 3, 1, 1, 1, 32),
    ((8, 4, 512), 8, 3, 1, 1, 1, 1),
    ((16, 24, 2048), 96, 7, 1, 3, 1, 2),
    ((8, 8, 8192), 16, 11, 4, 5, 1, 1),
    ((64, 48, 1024), 128, 5, 2, 2, 1, 1),
]

CONV2D_CORE_CASES = [
    ((16, 64, 56, 56), 128, 1, 1, 0, 1, 1),
    ((8, 256, 64, 64), 256, 3, 1, 1, 1, 1),
    ((16, 32, 32, 32), 32, 3, 1, 2, 2, 1),
]

CONV2D_MORE_CASES = [
    # asymmetric kernel / stride / padding / dilation
    ((16, 32, 32, 32), 32, (3, 5), (2, 1), (1, 2), (1, 2), 1),
    ((32, 64, 128, 128), 32, 3, 1, 2, 1, 1),
    ((16, 32, 56, 56), 32, 3, 1, 1, 1, 32),
    ((32, 64, 210, 210), 16, 5, 2, 1, 1, 1),
    ((16, 32, 24, 24), 24, 3, 2, 2, 1, 2),
    ((8, 3, 224, 224), 16, 3, 1, 1, 1, 1),
]

CONV3D_CORE_CASES = [
    ((2, 16, 12, 12, 12), 16, 3, 1, 1, 1, 16),
    ((2, 16, 16, 16, 16), 16, 3, 1, 1, 1, 1),
    ((2, 4, 16, 16, 16), 8, 3, 1, 1, 1, 1),
]

CONV3D_MORE_CASES = [
    ((4, 16, 24, 24, 24), 16, 3, 1, 1, 1, 1),
    ((2, 16, 16, 16, 16), 16, 3, 2, 1, 1, 1),
]


def _as_tuple(value, ndim):
    """Broadcast a scalar spatial parameter to one entry per dimension."""
    if isinstance(value, (tuple, list)):
        assert len(value) == ndim, f"expected {ndim} values, got {value}"
        return tuple(value)
    return (value,) * ndim


_CONV_BASELINE_FN = {
    1: torch.nn.functional.conv1d,
    2: torch.nn.functional.conv2d,
    3: torch.nn.functional.conv3d,
}


def conv_baseline(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    """Baseline for one case, with the same signature as the operator under test.

    The obvious baseline is torch.ops.aten.cudnn_convolution.default -- it is the
    exact operator being reimplemented -- but it is a low-level entry point that
    hands the spatial rank straight to cuDNN and never performs the 1D -> 2D
    promotion that at::_convolution does before dispatching. On thead's PPU
    runtime a 3-D input therefore arrives with 1-element padding/stride/dilation
    and faults inside the vendor convolution layer; the fault is raised as a
    signal rather than an exception, so it kills the pytest process outright on
    the first case instead of being reported as a failed case.

    torch.nn.functional.conv{1,2,3}d dispatches through at::_convolution, which
    promotes 1D to 2D first, and is what the sibling conv1d/conv2d/conv3d
    benchmarks use. On NVIDIA it still lands in cuDNN, so the comparison the
    benchmark is meant to make is unchanged. The three cuDNN flags are not part
    of the functional API and are ignored.
    """
    return _CONV_BASELINE_FN[input.ndim - 2](
        input,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def cudnn_convolution_input_fn(case, dtype, device):
    """Build the positional argument tuple for one benchmark case.

    Both the baseline (see conv_baseline) and flag_gems.cudnn_convolution take
    (input, weight, padding, stride, dilation, groups, benchmark, deterministic,
    allow_tf32), so the same argument list drives both. The last three flags are
    ignored by the Triton implementation.
    """
    in_shape, out_c, kernel, stride, padding, dilation, groups = case
    ndim = len(in_shape) - 2
    weight_shape = (out_c, in_shape[1] // groups, *_as_tuple(kernel, ndim))

    inp = utils.generate_tensor_input(in_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)

    yield (
        inp,
        weight,
        list(_as_tuple(padding, ndim)),
        list(_as_tuple(stride, ndim)),
        list(_as_tuple(dilation, ndim)),
        groups,
        False,  # benchmark
        False,  # deterministic
        False,  # allow_tf32
    )


CONV_CORE_CASES = CONV1D_CORE_CASES + CONV2D_CORE_CASES + CONV3D_CORE_CASES
CONV_MORE_CASES = CONV1D_MORE_CASES + CONV2D_MORE_CASES + CONV3D_MORE_CASES


class CudnnConvBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        cases = list(CONV_CORE_CASES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            cases += CONV_MORE_CASES
        self.shapes = list(dict.fromkeys(cases))

    def get_input_iter(self, dtype) -> Generator:
        for case in self.shapes:
            yield from self.input_fn(case, dtype, self.device)


@pytest.mark.cudnn_convolution
def test_cudnn_convolution():
    torch.backends.cudnn.allow_tf32 = False
    bench = CudnnConvBenchmark(
        input_fn=cudnn_convolution_input_fn,
        op_name="cudnn_convolution",
        torch_op=conv_baseline,
        dtypes=BENCH_DTYPES,
        gems_op=flag_gems.cudnn_convolution,
    )
    bench.run()
