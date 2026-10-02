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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# aten::resize_as_sparse_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)
# Resizes a sparse COO tensor in place to the template's shape and split, so the
# cost tracks nnz plus the sparse/dense storage touched. Dedicated descriptors:
# (self_shape, self_sparse_dim, nnz, coalesced, template_shape,
# template_sparse_dim). The template never shrinks a non-empty self, which the
# reference rejects.
_SPARSE_RESIZE_CASES = [
    ((1024, 1024), 2, 65536, True, (1024, 1024), 2),
    ((1024, 1024), 2, 65536, True, (2048, 1024), 2),
    ((1024, 1024), 2, 1048576, True, (1024, 2048), 2),
    ((4096, 4096), 2, 1048576, True, (4096, 8192), 2),
    ((64, 512, 512), 3, 524288, True, (64, 512, 1024), 3),
    ((16, 1024, 1024), 2, 8192, True, (32, 1024, 1024), 2),
    ((1024, 1024), 2, 65536, False, (2048, 1024), 2),
    ((1024, 1024), 2, 0, True, (2048, 2048), 2),
]


def _descriptor_from_shape(shape):
    """Turn a framework-requested dense shape into a sparse resize descriptor.

    Every requested extent is preserved as ``self_shape``. The split is
    all-sparse so ``nnz`` values cost 1/1024 of the requested volume instead of
    a dense tail that would dominate the measurement, and the template grows the
    outer extent by one, which the reference accepts for a non-empty self.
    """
    if len(shape) == 6 and isinstance(shape[0], (tuple, list)):
        return tuple(tuple(part) if isinstance(part, list) else part for part in shape)
    shape = tuple(shape)
    space = math.prod(shape) if shape else 1
    if space <= 1:
        # Nothing to grow: the template repeats the requested extents.
        return (shape, len(shape), 0, True, shape, len(shape))
    nnz = max(1, space // 1024)
    return (shape, len(shape), nnz, True, (shape[0] + 1,) + shape[1:], len(shape))


def _unique_flat(count, space, generator):
    """``count`` distinct coordinates drawn from ``[0, space)``."""
    if count > space:
        raise ValueError("unique stored entries exceed the sparse coordinate space")
    if count == space:
        return torch.arange(space, dtype=torch.long)
    if space <= 4 * count:
        return torch.sort(torch.randperm(space, generator=generator)[:count]).values
    # A strided draw is unique by construction and never allocates the whole
    # coordinate space.
    return torch.arange(count, dtype=torch.long) * (space // count)


def _make_sparse_input(shape, sparse_dim, nnz, coalesced, dtype, device):
    dense_shape = tuple(shape[sparse_dim:])
    if nnz == 0:
        indices = torch.empty((sparse_dim, 0), dtype=torch.long, device=device)
        values = torch.empty((0,) + dense_shape, dtype=dtype, device=device)
    else:
        generator = torch.Generator("cpu").manual_seed(0)
        flat = _unique_flat(
            nnz if coalesced or nnz == 1 else nnz - 1,
            math.prod(shape[:sparse_dim]),
            generator,
        )
        if not coalesced and nnz > 1:
            # Repeating the first coordinate keeps the tensor non-coalesced.
            flat = torch.cat([flat[:1], flat])
        indices = torch.stack(torch.unravel_index(flat, shape[:sparse_dim]), dim=0).to(
            device
        )
        values = torch.randn((nnz,) + dense_shape, device=device).to(dtype)
    return torch.sparse_coo_tensor(
        indices, values, shape, device=device, is_coalesced=coalesced
    )


def _case_fn(shape, dtype):
    # Every entry of self.shapes is a descriptor, so the plan is metadata only:
    # no tensor is allocated while cases are collected or listed.
    del dtype
    (
        self_shape,
        self_sparse_dim,
        nnz,
        coalesced,
        template_shape,
        template_sparse_dim,
    ) = shape
    yield base.BenchmarkCasePlan(
        shape={"input": list(self_shape), "template": list(template_shape)},
        params={
            "sparse_dim": self_sparse_dim,
            "nnz": nnz,
            "coalesced": coalesced,
            "template_sparse_dim": template_sparse_dim,
        },
        builder_args=shape,
    )


def _build_inputs_fn(plan, dtype, device):
    (
        self_shape,
        self_sparse_dim,
        nnz,
        coalesced,
        template_shape,
        template_sparse_dim,
    ) = plan.builder_args
    inp = _make_sparse_input(self_shape, self_sparse_dim, nnz, coalesced, dtype, device)
    # The template only supplies sizes and the split; it stays empty.
    template = torch.sparse_coo_tensor(
        torch.empty((template_sparse_dim, 0), dtype=torch.long, device=device),
        torch.empty(
            (0,) + tuple(template_shape[template_sparse_dim:]),
            dtype=dtype,
            device=device,
        ),
        template_shape,
        device=device,
        is_coalesced=True,
    )
    return inp, template


class ResizeAsSparseBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Follow the normal framework resolution (core_shapes.yaml, the class
        # MRO or a user --shape_file), then UNION the dedicated sparse
        # descriptors, so every requested extent is kept and no case repeats.
        super().set_shapes(shape_file_path)
        shapes = [_descriptor_from_shape(shape) for shape in self.shapes]
        shapes.extend(
            descriptor
            for descriptor in _SPARSE_RESIZE_CASES
            if descriptor not in shapes
        )
        self.shapes = shapes


@pytest.mark.resize_as_sparse_
def test_resize_as_sparse_():
    bench = ResizeAsSparseBenchmark(
        op_name="resize_as_sparse_",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.resize_as_sparse_,
        gems_op=getattr(flag_gems, "resize_as_sparse_", None),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
