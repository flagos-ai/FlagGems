# Copyright 2026, The FlagGems Authors.
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

"""Benchmark for ``aten::mkldnn_max_pool2d``.

oneDNN-only, so both the reference and the candidate receive the operator's real
argument type: a CPU oneDNN tensor built directly on the CPU. The builder returns
the arguments flat (input, kernel_size, stride, padding, dilation, ceil_mode)
with a trailing kwargs dict, which is what ``unpack_to_args_kwargs`` expects.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# ResNet-style activation shapes.
_BENCH_SHAPES = [
    (4, 3, 224, 224),
    (16, 64, 56, 56),
    (32, 128, 28, 28),
    (64, 256, 14, 14),
    (128, 512, 7, 7),
]

# dilation stays 1: the oracle rejects any other value with
# "mkldnn_max_pool2d does not support dilation case".
_CORE = ([3, 3], [2, 2], [1, 1], [1, 1], False)
_NON_SQUARE = ([3, 5], [2, 1], [1, 2], [1, 1], False)
_CEIL = ([3, 3], [2, 2], [1, 1], [1, 1], True)
_K2S2 = ([2, 2], [2, 2], [0, 0], [1, 1], False)

# Extended parameter rows are attached to the shapes they are meaningful for; a
# shape supplied through --shape_file falls back to the core row instead of
# being dropped.
_BENCH_PARAMS = {
    (4, 3, 224, 224): (_CORE, _NON_SQUARE),
    (16, 64, 56, 56): (_CORE, _NON_SQUARE, _CEIL),
    (32, 128, 28, 28): (_CORE, _K2S2),
    (64, 256, 14, 14): (_CORE,),
    (128, 512, 7, 7): (_CORE,),
}

# oneDNN holds these element types only; float64/int32/int64/fp8 operands cannot
# be constructed ("dense_to_mkldnn expects float, bfloat16, half, uint8, int8").
_BENCH_DTYPES = consts.FLOAT_DTYPES + [torch.int8, torch.uint8]


def _case_params(params):
    """JSON-compatible copy of one parameter row (listing stays metadata-only)."""
    return {
        "kernel_size": list(params[0]),
        "stride": list(params[1]),
        "padding": list(params[2]),
        "dilation": list(params[3]),
        "ceil_mode": bool(params[4]),
    }


def _case_fn(shape, dtype):
    """Case metadata only -- no tensor is allocated while listing cases."""
    del dtype
    for params in _BENCH_PARAMS.get(shape, (_CORE,)):
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params=_case_params(params),
            builder_args=(shape,),
        )


def _build_inputs_fn(plan, dtype, device):
    del device  # oneDNN operands exist only on CPU.
    shape = plan.builder_args[0]
    # Direct CPU allocation: generate_tensor_input has no branch for int8/uint8.
    inp = torch.empty(shape, dtype=dtype, device="cpu").to_mkldnn()
    params = plan.params
    # Flat positional arguments plus a trailing kwargs dict.
    return (
        inp,
        params["kernel_size"],
        params["stride"],
        params["padding"],
        params["dilation"],
        params["ceil_mode"],
        {},
    )


class MkldnnMaxPool2dBenchmark(OperatorBenchmark):
    """Pooling only accepts rank-4 (N, C, H, W) operands."""

    def set_shapes(self, shape_file_path=None):
        # The ordinary super() call keeps --shape_file support and the shared
        # grids; the operator's static rank-4 legality drops only rows that
        # cannot be operands, and the native-valid rows are unioned back in.
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(
                [tuple(shape) for shape in self.shapes if len(tuple(shape)) == 4]
                + _BENCH_SHAPES
            )
        )


@pytest.mark.mkldnn_max_pool2d
def test_mkldnn_max_pool2d():
    bench = MkldnnMaxPool2dBenchmark(
        op_name="mkldnn_max_pool2d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_max_pool2d,
        gems_op=getattr(flag_gems, "mkldnn_max_pool2d", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
