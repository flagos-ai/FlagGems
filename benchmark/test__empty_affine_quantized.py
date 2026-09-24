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

# ``_empty_affine_quantized`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register the
# markers directly on the MarkGenerator so ``@pytest.mark._empty_affine_quantized``
# and ``-m _empty_affine_quantized`` both work.
for _name in ("_empty_affine_quantized", "_empty_affine_quantized_out"):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# aten::_empty_affine_quantized is a factory that returns a fresh per-tensor
# affine quantized tensor with uninitialized storage, so the benchmark measures
# dispatch + storage-construction overhead rather than memory bandwidth. The
# default shape set contains a 1-B-element 1-D tensor whose cost would be
# dominated by input allocation; use allocation-friendly shapes that still
# exercise a realistic range of ranks (1-D through 4-D, including the canonical
# (1024, 1024) and (20, 320, 15) shapes from the shared constants).
EMPTY_AFFINE_QUANTIZED_SHAPES = [
    (1024,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (20, 320, 15),
    (16, 128, 64, 1280),
]

# Quantized storage dtypes; the .out variant exercises the qparam-reset path in
# addition to the fill/construction work.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# Representative per-tensor affine qparams (a positive float scale and a small
# integer zero_point), matching the values used by the correctness tests.
SCALE = 0.1
ZERO_POINT = 0


def _case_fn(shape, dtype):
    # The case list is orthogonal to dtype: one case per shape.
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"size": shape},
        params={"scale": SCALE, "zero_point": ZERO_POINT},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    # size is the only positional argument; dtype/scale/zero_point/device are
    # keyword-only on the aten factory, so they must travel through kwargs.
    shape = plan.builder_args[0]
    return shape, {
        "dtype": dtype,
        "scale": plan.params["scale"],
        "zero_point": plan.params["zero_point"],
        "device": device,
    }


def _build_inputs_fn_out(plan, dtype, device):
    # The .out variant writes into (and returns) the provided buffer without
    # changing its dtype, so the buffer is created with the benchmarked
    # quantized dtype.
    shape = plan.builder_args[0]
    out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=dtype, device=device, scale=1.0, zero_point=0
    )
    return shape, {
        "scale": plan.params["scale"],
        "zero_point": plan.params["zero_point"],
        "out": out,
    }


class EmptyAffineQuantizedBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes.

    The default shape set contains a 1-B-element 1-D tensor whose cost would be
    dominated by input allocation, so the case list is restricted to the shapes
    above.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(
            shape_file_path, default_shapes=EMPTY_AFFINE_QUANTIZED_SHAPES
        )


@pytest.mark._empty_affine_quantized
def test__empty_affine_quantized():
    bench = EmptyAffineQuantizedBenchmark(
        op_name="_empty_affine_quantized",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._empty_affine_quantized,
        # The candidate defaults to the process-local KernelGen override and is
        # resolved inside Benchmark.run() via Benchmark._candidate_call; None is the
        # fallback until flag_gems registers the operator.
        gems_op=getattr(flag_gems, "_empty_affine_quantized", None),
        dtypes=QUANT_DTYPES,
    )
    bench.run()


@pytest.mark._empty_affine_quantized_out
def test__empty_affine_quantized_out():
    bench = EmptyAffineQuantizedBenchmark(
        op_name="_empty_affine_quantized",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten._empty_affine_quantized.out,
        gems_op=getattr(flag_gems, "_empty_affine_quantized", None),
        dtypes=QUANT_DTYPES,
    )
    bench.run()
