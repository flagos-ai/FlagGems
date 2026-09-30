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

from . import base
from .generated_operator_utils import OperatorBenchmark

# aten::qscheme(Tensor self) -> QScheme is a metadata-only query: it reports the
# saved quantizer and reads no payload, so no public Benchmark family covers it
# and the two-phase case_fn/build_inputs_fn API below keeps case listing
# tensor-free. The quantized element types are the operator's whole dtype set;
# every other dtype has no native kernel.
_PER_TENSOR = 0
_PER_CHANNEL = 1
_PER_CHANNEL_FLOAT_QPARAMS = 4
_SCHEME_NAMES = {
    _PER_TENSOR: "per_tensor_affine",
    _PER_CHANNEL: "per_channel_affine",
    _PER_CHANNEL_FLOAT_QPARAMS: "per_channel_affine_float_qparams",
}

_QSCHEME_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# This operator's own grid, merged at comprehensive level on top of the shared
# core_shapes grid; core keeps that shared grid. The 0-dim entry is the scalar
# per-tensor form, and per-channel schemes -- which need a channel axis -- are
# skipped for it.
_EXTRA_SHAPES = [
    (),
    (16384,),
    (64, 64),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64),
    (16, 128, 64, 60),
    (16, 7, 57, 32),
]


def _case_fn(shape, dtype):
    del dtype
    # Only metadata is emitted here: listing must not allocate tensor inputs.
    yield base.BenchmarkCasePlan(
        shape={"self": shape},
        params={"scheme": _SCHEME_NAMES[_PER_TENSOR]},
        builder_args=(shape, _PER_TENSOR),
    )
    if len(shape) < 1:
        return
    for scheme in (_PER_CHANNEL, _PER_CHANNEL_FLOAT_QPARAMS):
        yield base.BenchmarkCasePlan(
            shape={"self": shape},
            params={"scheme": _SCHEME_NAMES[scheme]},
            builder_args=(shape, scheme),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, scheme = plan.builder_args
    if scheme == _PER_TENSOR:
        # Uninitialized quantized payload: the query never reads it.
        return (
            torch._empty_affine_quantized(
                list(shape), scale=0.1, zero_point=10, dtype=dtype, device=device
            ),
        )
    channels = shape[0]
    # The native constructor stores contiguous qparams with numel == size(axis),
    # so the expanded source costs nothing beyond one shared element.
    scales = torch.full((1,), 0.25, dtype=torch.float64, device=device).expand(channels)
    if scheme == _PER_CHANNEL:
        zero_points = torch.full((1,), 10, dtype=torch.int64, device=device).expand(
            channels
        )
    else:
        zero_points = torch.full((1,), 0.5, dtype=torch.float64, device=device).expand(
            channels
        )
    return (
        torch._empty_per_channel_affine_quantized(
            list(shape),
            scales=scales,
            zero_points=zero_points,
            axis=0,
            dtype=dtype,
            device=device,
        ),
    )


class _QSchemeBenchmark(OperatorBenchmark):
    """Two-phase benchmark for a metadata-only query."""

    def set_more_shapes(self):
        # Comprehensive level: the shared compatible extras plus this operator's
        # own shapes (deduplicated by the base implementation).
        return super().set_more_shapes() + list(_EXTRA_SHAPES)


@pytest.mark.qscheme
def test_qscheme():
    bench = _QSchemeBenchmark(
        op_name="qscheme",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.qscheme,
        gems_op=getattr(flag_gems, "qscheme", None),
        dtypes=_QSCHEME_DTYPES,
    )
    bench.run()
