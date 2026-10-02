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

import flag_gems

from . import base

# aten::q_per_channel_axis reads a per-channel quantizer's stored channel axis and
# returns a Python int. A case is a BenchmarkCasePlan whose shape carries the
# operand geometry and whose params carry the saved axis, so --list-cases
# allocates no tensor and executes no operator while execution rebuilds exactly
# the same case from builder_args. Input allocation happens outside timing.

# Quantized element dtypes. The correctness file also covers the sub-byte types;
# this trio matches the other quantized benchmarks in this directory.
QUANT_DTYPES = [torch.qint8, torch.quint8, torch.qint32]


def _axes_for(shape):
    # The factory stores the channel axis verbatim, so the plans cover the first,
    # middle and last axis plus the negative spelling of the last axis. A 0-dim
    # operand has a single channel, making 0 its only stored axis.
    if len(shape) == 0:
        return (0,)
    axes = [0, len(shape) // 2, len(shape) - 1, -1]
    return tuple(dict.fromkeys(axes))


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    for axis in _axes_for(shape):
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"axis": axis},
            builder_args=(shape, axis),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, axis = plan.builder_args
    # A 0-dim operand carries one channel, the same factory contract the
    # correctness file uses; any other shape is sized by its stored axis.
    channels = 1 if len(shape) == 0 else shape[axis]
    scales = torch.ones(channels, dtype=torch.float64, device=device)
    zero_points = torch.zeros(channels, dtype=torch.int64, device=device)
    # scales/zero_points/axis/dtype/device are keyword-only on the aten factory,
    # so they travel in the trailing kwargs dict while the operand is the only
    # positional argument: op(input).
    inp = torch.ops.aten._empty_per_channel_affine_quantized(
        list(shape),
        scales=scales,
        zero_points=zero_points,
        axis=axis,
        dtype=dtype,
        device=device,
    )
    return inp, {}


@pytest.mark.q_per_channel_axis
def test_q_per_channel_axis():
    bench = base.GenericBenchmark(
        op_name="q_per_channel_axis",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.q_per_channel_axis,
        gems_op=getattr(flag_gems, "q_per_channel_axis", None),
        dtypes=QUANT_DTYPES,
    )
    bench.run()
