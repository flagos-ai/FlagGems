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

import math

import pytest
import torch

import flag_gems

from . import base, utils

# aten::_nnpack_spatial_convolution is the NNPACK CPU engine: it takes float32
# NCHW CPU tensors only and rejects accelerator tensors and other dtypes with
# "Mismatched Tensor types in NNPack convolutionOutput", so the reference, the
# candidate inputs and the timing all stay on the CPU. The static build flag allocates no tensor and calls no operator.
_CPU = torch.device("cpu")
_DTYPES = [torch.float32]
_NNPACK_AVAILABLE = "USE_NNPACK=ON" in torch.__config__.show()


# Original NCHW shapes, kept unchanged. A caller-supplied shape file still
# overrides them through the shared loader; native execution is 4-D only.
_DEFAULT_SHAPES = [
    (1, 3, 224, 224),
    (1, 16, 32, 32),
    (8, 32, 28, 28),
    (16, 64, 64, 64),
]

# Additional NCHW examples supplement the shared grid.
_MORE_SHAPES = [(2, 4, 9, 7), (1, 8, 32, 32)]

# Native-valid (kernel, padding, stride) variants from the original suite.
_KERNEL_PAD_STRIDE = [
    (3, [1, 1], [1, 1]),
    (5, [2, 2], [1, 1]),
    (3, [1, 1], [2, 2]),
]


class NnpackSpatialConvolutionBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _DEFAULT_SHAPES
    DEFAULT_SHAPE_DESC = "N, C, H, W"

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(dict.fromkeys(_nchw_shape(shape) for shape in self.shapes))
        for shape in _DEFAULT_SHAPES:
            if shape not in self.shapes:
                self.shapes.append(shape)

    def set_more_shapes(self):
        return list(super().set_more_shapes()) + _MORE_SHAPES


def _nchw_shape(shape):
    # Bare shapes describe input extents. Preserve every element while supplying
    # the NCHW axes required by this CPU primitive.
    shape = tuple(shape)
    if len(shape) == 1:
        width = math.isqrt(shape[0])
        while width > 1 and shape[0] % width:
            width -= 1
        return (1, 1, width, shape[0] // width) if width else (1, 1, 0, 0)
    if len(shape) < 4:
        return (1,) * (4 - len(shape)) + shape
    return (math.prod(shape[:-3]),) + shape[-3:]


def _case_fn(shape, dtype):
    """One metadata plan per kernel, padding and stride variant."""
    del dtype
    in_channels = shape[1] if len(shape) == 4 else 0
    out_channels = min(64, max(4, in_channels * 2))
    for kernel, padding, stride in _KERNEL_PAD_STRIDE:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={
                "shape_rank": len(shape),
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel": kernel,
                "padding": str(padding),
                "stride": str(stride),
            },
            builder_args=(shape, out_channels, kernel, padding, stride),
        )


def _build_inputs_fn(plan, dtype, device):
    del device
    torch.backends.nnpack.is_available()
    shape, out_channels, kernel, padding, stride = plan.builder_args
    if len(shape) != 4:
        raise ValueError(
            "aten::_nnpack_spatial_convolution needs 4-D NCHW operands; got "
            + repr(shape)
        )
    inp = utils.generate_tensor_input(shape, dtype, _CPU)
    weight = utils.generate_tensor_input(
        (out_channels, shape[1], kernel, kernel), dtype, _CPU
    )
    bias = utils.generate_tensor_input((out_channels,), dtype, _CPU)
    return inp, weight, bias, padding, stride


@pytest.mark.nnpack_spatial_convolution
@pytest.mark.skipif(
    not _NNPACK_AVAILABLE,
    reason="PyTorch was built without NNPACK",
)
def test__nnpack_spatial_convolution():
    bench = NnpackSpatialConvolutionBenchmark(
        op_name="_nnpack_spatial_convolution",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._nnpack_spatial_convolution,
        gems_op=getattr(flag_gems, "_nnpack_spatial_convolution", None),
        dtypes=_DTYPES,
    )
    bench.run()
