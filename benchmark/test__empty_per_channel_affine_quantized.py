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

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base

# ``_empty_per_channel_affine_quantized`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register the markers directly on the MarkGenerator so
# ``@pytest.mark._empty_per_channel_affine_quantized`` and
# ``-m _empty_per_channel_affine_quantized`` both work.
for _name in (
    "_empty_per_channel_affine_quantized",
    "_empty_per_channel_affine_quantized_out",
):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# aten::_empty_per_channel_affine_quantized is a quantized-tensor factory whose
# cost is dominated by storage allocation. The default shape set contains a
# 1-B-element 1-D tensor whose allocation would dominate the measurement, so the
# case list is restricted to allocation-friendly shapes that still exercise a
# realistic range of ranks (1-D through 4-D, including the canonical
# (1024, 1024) and (20, 320, 15) shapes from the shared constants).
EMPTY_Q_SHAPES = [
    (1024,),
    (64, 64),
    (1024, 1024),
    (20, 320, 15),
    (64, 512, 512),
    (16, 128, 64, 1280),
]

# Quantized element dtypes. The full set lives in the correctness tests; the
# benchmark uses the common trio to keep the timing grid focused.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]


def _make_metadata(shape, dtype, device):
    # axis=0 for every case keeps the case matrix orthogonal to the shape
    # geometry. Positive zero_points avoid the CUDA per-channel affine quantizer
    # lower-bound validation.
    del dtype
    num_channels = shape[0]
    scales = torch.arange(1, num_channels + 1, dtype=torch.float64, device=device)
    zero_points = torch.randint(0, 8, (num_channels,), dtype=torch.int64, device=device)
    return scales, zero_points


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"size": shape},
        params={"axis": 0},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    scales, zero_points = _make_metadata(shape, dtype, device)
    # scales/zero_points/axis/dtype/device are keyword-only on the aten factory,
    # so they must be delivered through the kwargs dict.
    return shape, {
        "scales": scales,
        "zero_points": zero_points,
        "axis": 0,
        "dtype": dtype,
        "device": device,
    }


def _build_inputs_fn_out(plan, dtype, device):
    shape = plan.builder_args[0]
    scales, zero_points = _make_metadata(shape, dtype, device)
    # The .out variant writes into (and returns) the provided buffer without
    # changing its dtype, so the buffer is created with the benchmarked
    # quantized dtype.
    out = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=torch.tensor([1.0], dtype=torch.float64, device=device),
        zero_points=torch.tensor([0], dtype=torch.int64, device=device),
        axis=0,
        dtype=dtype,
        device=device,
    )
    return shape, {
        "scales": scales,
        "zero_points": zero_points,
        "axis": 0,
        "out": out,
    }


class EmptyPerChannelAffineQuantizedBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes.

    ``case_fn`` + ``build_inputs_fn`` drive the per-case materialization; the
    candidate is the process-local KernelGen override, resolved inside
    ``Benchmark.run()`` through ``Benchmark._candidate_call`` (the
    ``gems_op`` fallback below is only used once flag_gems registers the op).
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=EMPTY_Q_SHAPES)


@pytest.mark._empty_per_channel_affine_quantized
def test__empty_per_channel_affine_quantized():
    bench = EmptyPerChannelAffineQuantizedBenchmark(
        op_name="_empty_per_channel_affine_quantized",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._empty_per_channel_affine_quantized,
        gems_op=getattr(flag_gems, "_empty_per_channel_affine_quantized", None),
        dtypes=QUANT_DTYPES,
    )
    bench.run()


@pytest.mark._empty_per_channel_affine_quantized_out
def test__empty_per_channel_affine_quantized_out():
    bench = EmptyPerChannelAffineQuantizedBenchmark(
        op_name="_empty_per_channel_affine_quantized",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten._empty_per_channel_affine_quantized.out,
        gems_op=getattr(flag_gems, "_empty_per_channel_affine_quantized", None),
        dtypes=QUANT_DTYPES,
    )
    bench.run()
