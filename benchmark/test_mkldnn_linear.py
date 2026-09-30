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

import pytest
import torch

import flag_gems

from . import base, consts

# aten::mkldnn_linear is a CPU-only oneDNN inner product: `self` is an mkldnn
# tensor and `weight`/`bias` are dense, so every operand is allocated on CPU even
# though the harness device default is flag_gems.device.
MKLDNN_LINEAR_SHAPES = [
    (64, 512),
    (256, 1024),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
]

# Preserved operator configuration: explicit out_features per shape, 256 by
# default.
_OUT_FEATURES = {(16, 128, 64, 60): 64}
_DEFAULT_OUT_FEATURES = 256

# Rank-1 rows of the shared shape sets are natively legal for `self`, but keeping
# the 2**30 / 2**28 rows as vectors would require a (out_features, 2**30) weight.
# This numel-preserving (rows, 1024) geometry keeps the original element count
# and a meaningful feature width; every other shared row is used as configured.
_RANK1_RESHAPE = {
    (1024 * 1024 * 1024,): (1024 * 1024, 1024),
    (2**28,): (2**18, 1024),
}

# oneDNN accepts exactly these CPU dtypes; to_mkldnn() rejects float64/int32/
# int64/bool and fp8 (dense_to_mkldnn expects float, bfloat16, half, uint8, int8
# tensor input).
BENCH_DTYPES = consts.FLOAT_DTYPES + [torch.int8, torch.uint8]

# The oneDNN inner product has no int8/uint8 bias path.
_DTYPES_WITHOUT_BIAS = (torch.int8, torch.uint8)


def _native_geometry(shapes, extra_shapes):
    # Normalize the shared rows to natively valid geometry and union them with the
    # operator's own rows. Rank-0 rows are dropped: an mkldnn operand cannot
    # describe a 0-dim activation, so the native op has no valid argument there.
    merged = []
    for shape in list(shapes) + list(extra_shapes):
        shape = tuple(shape)
        if shape:
            merged.append(_RANK1_RESHAPE.get(shape, shape))
    return list(dict.fromkeys(merged))


def _case_fn(shape, dtype):
    shape = tuple(shape)
    out_features = _OUT_FEATURES.get(shape, _DEFAULT_OUT_FEATURES)
    bias_modes = (False,) if dtype in _DTYPES_WITHOUT_BIAS else (True, False)
    for with_bias in bias_modes:
        yield base.BenchmarkCasePlan(
            shape={
                "input": shape,
                "weight": (out_features, shape[-1]),
                "bias": (out_features,) if with_bias else None,
            },
            params={"out_features": out_features, "bias": with_bias},
            builder_args=(shape, out_features, with_bias),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, out_features, with_bias = plan.builder_args
    # mkldnn operands cannot live on the harness device, so the CPU allocation
    # below is the native contract of this CPU-only operator.
    del device
    activation = torch.empty(shape, dtype=dtype).to_mkldnn()
    weight = torch.empty((out_features, shape[-1]), dtype=dtype)
    bias = torch.empty((out_features,), dtype=dtype) if with_bias else None
    # Flat positional operands plus a trailing kwargs dict: the shared unpacker
    # does not flatten a nested (args, kwargs) pair.
    return activation, weight, bias, {}


class MkldnnLinearBenchmark(base.GenericBenchmark):
    # GenericBenchmark over natively valid mkldnn inner-product geometry.

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Everything the base resolved (core defaults, the COMPREHENSIVE extras
        # or a caller's shape file) is kept; only the geometry is normalized and
        # the operator's own rows are added.
        self.shapes = _native_geometry(self.shapes, MKLDNN_LINEAR_SHAPES)


@pytest.mark.mkldnn_linear
def test_mkldnn_linear():
    bench = MkldnnLinearBenchmark(
        op_name="mkldnn_linear",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_linear,
        gems_op=getattr(flag_gems, "mkldnn_linear", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
