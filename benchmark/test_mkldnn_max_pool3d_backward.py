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

"""Benchmark for aten::mkldnn_max_pool3d_backward.

The operands are 5-D NCDHW MkldnnCPU tensors, so the shared loader's grid is
normalised to this operator's rank-5 workloads. _case_fn yields JSON-compatible
plans only, keeping --list-cases tensor-free and letting --case-id rebuild the
same operands. _build_inputs_fn follows unpack_to_args_kwargs: positional
arguments plus a trailing kwargs dict, with the native forward's own output kept
uncloned because it carries the argmax workspace the backward consumes.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# The operator's original verified NCDHW workloads.
_NCDHW_SHAPES = [
    (1, 2, 8, 9, 10),
    (2, 3, 16, 17, 18),
    (2, 5, 9, 10, 11),
    (16, 7, 57, 32, 29),
    (8, 16, 32, 32, 32),
]

# Additional rank-5 workloads merged in at comprehensive level.
_MORE_SHAPES = [
    (1, 1, 16, 16, 16),
    (4, 8, 16, 16, 16),
    (1, 1, 256, 256, 256),
]

# Small descriptor that carries every cheap parameter branch.
_SMALL_DESCRIPTOR = (1, 2, 8, 9, 10)

# (kernel_size, stride, padding, ceil_mode); dilation stays at 1 and stride stays
# positive (stride 0 crashes the vendor kernel with SIGFPE).
_CORE_ROWS = [
    ((2, 2, 2), (2, 2, 2), (0, 0, 0), False),
    ((3, 3, 3), (1, 1, 1), (0, 0, 0), False),
    ((3, 3, 3), (3, 3, 3), (0, 0, 0), True),
]
_EXTRA_ROWS = [
    ((1, 1, 1), (1, 1, 1), (0, 0, 0), False),
    ((2, 3, 4), (2, 3, 4), (0, 0, 0), False),
    ((2, 2, 2), (2, 2, 2), (1, 1, 1), False),
    ((3, 3, 3), (3, 3, 3), (1, 1, 1), True),
    ((3, 3, 3), (3, 3, 3), (0, 0, 0), False),
]


def _checked_shape(shape):
    extents = tuple(shape)
    if len(extents) != 5:
        raise ValueError(
            "mkldnn_max_pool3d requires a 5-D NCDHW shape, got " + repr(extents)
        )
    for extent in extents:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 1:
            raise ValueError("invalid NCDHW extent " + repr(extent))
    return extents


def _plan(shape, kernel_size, stride, padding, ceil_mode):
    return base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={
            "kernel_size": list(kernel_size),
            "stride": list(stride),
            "padding": list(padding),
            "ceil_mode": ceil_mode,
        },
        builder_args=(
            tuple(shape),
            tuple(kernel_size),
            tuple(stride),
            tuple(padding),
            ceil_mode,
        ),
    )


def _case_fn(shape, dtype):
    del dtype  # the dtype comes from the benchmark's own dtype loop
    shape = _checked_shape(shape)
    # The small descriptor carries every cheap parameter branch; larger
    # descriptors carry the core rows only.
    rows = list(_CORE_ROWS)
    if shape == _SMALL_DESCRIPTOR:
        rows += _EXTRA_ROWS
    for kernel_size, stride, padding, ceil_mode in rows:
        yield _plan(shape, kernel_size, stride, padding, ceil_mode)


def _build_inputs_fn(plan, dtype, device):
    del device  # MkldnnCPU operands exist on the CPU only.
    shape, kernel_size, stride, padding, ceil_mode = plan.builder_args
    # Dense operands are built on the CPU directly: a device round trip would only
    # add traffic for an operand that has to be a CPU oneDNN tensor. The gradient
    # is handed over untouched - the framework keeps is_backward at its default,
    # and a cloned oneDNN tensor makes the primitive reorder fail.
    dense = torch.randn(shape, dtype=dtype, device="cpu")
    # requires_grad makes the forward save the argmax workspace the backward
    # re-creates; dilation is passed at its only supported value.
    forward_input = dense.detach().clone().requires_grad_(True)
    output = torch.ops.aten.mkldnn_max_pool3d(
        forward_input.to_mkldnn(),
        list(kernel_size),
        list(stride),
        list(padding),
        [1, 1, 1],
        ceil_mode,
    )
    # grad_output must match the forward output's shape; the backward cannot
    # reorder a gradient shaped like the input into the pooling layout.
    grad_output = torch.randn(tuple(output.shape), dtype=dtype, device="cpu")
    return grad_output.to_mkldnn(), {
        "output": output,
        "input": forward_input.to_mkldnn(),
        "kernel_size": list(kernel_size),
        "stride": list(stride),
        "padding": list(padding),
        "dilation": [1, 1, 1],
        "ceil_mode": ceil_mode,
    }


class MkldnnMaxPool3dBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # The normal shared loader runs first, so caller shape files keep working
        # and the comprehensive-level set_more_shapes merge still happens.
        super().set_shapes(shape_file_path)
        # The loader's generic grid entries are rank-1 and rank-2 tensors. They
        # are statically inapplicable here: mkldnn_max_pool3d derives the pooling
        # dimension from the kernel length and rejects any rank other than 5, so
        # no rank-1/2 tensor can ever be an operand for this operator. Every
        # loaded rank-5 descriptor is kept and the operator's own verified NCDHW
        # workloads are unioned in; no native-valid rank-5 descriptor is dropped or resized.
        loaded = [
            extents
            for extents in (tuple(shape) for shape in self.shapes)
            if len(extents) == 5
        ]
        self.shapes = list(dict.fromkeys(loaded + _NCDHW_SHAPES))

    def set_more_shapes(self):
        # Merged by the shared loader at comprehensive level; rank-5 as well.
        return list(_MORE_SHAPES)


@pytest.mark.mkldnn_max_pool3d_backward
def test_mkldnn_max_pool3d_backward():
    bench = MkldnnMaxPool3dBackwardBenchmark(
        op_name="mkldnn_max_pool3d_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_max_pool3d_backward,
        gems_op=getattr(flag_gems, "mkldnn_max_pool3d_backward", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
