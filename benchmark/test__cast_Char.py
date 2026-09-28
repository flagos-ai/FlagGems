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

"""Benchmarks for ``aten::_cast_Char(self, non_blocking=False) -> Tensor``.

The unary pointwise family supplies the shape scales (including the
``core_shapes.yaml`` class lookup and ``--shape_file``), the metrics and the
level handling. Case plans hold metadata only, so ``--list-cases`` allocates no
tensor, and the same plans drive execution and ``--case-id`` replay.

Every listed shape is the shape its builder delivers: the call family yields the
contiguous tensor, optionally with the ``non_blocking`` keyword, and the view
family yields a transposed, channels-last or row-padded slice with the listed
extents.

Only ``consts.FLOAT_DTYPES`` are benchmarked, because
``utils.generate_tensor_input`` has no int8/uint8/int64 generator; bfloat16 is
kept only when the device advertises it.
"""

import pytest
import torch

import flag_gems

from . import base, consts, utils

_NON_BLOCKING_FORMS = (("omitted", None), ("false", False), ("true", True))

# Extra 4-D shapes so the comprehensive level also covers the channels-last
# layout; the unary pointwise scales themselves stay untouched.
_VIEW_EXTRA_SHAPES = ((256, 256, 4, 4), (8, 64, 128, 128))


def _benchmark_dtypes():
    """The supported float dtypes, gated by the static device capability.

    The same function produces the dtype list for listing and for execution, so
    both see exactly the same cases.
    """
    return [
        dtype
        for dtype in consts.FLOAT_DTYPES
        if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
    ]


def _validated_shape(shape):
    """Normalize one shape, rejecting malformed extents before a plan exists.

    Every shape - the class scales, the extras added here and any supplied
    through ``--shape_file`` - goes through this check, so a bad extent fails the
    run instead of producing a plan that cannot be built. Scalar and zero
    extents are valid and are kept.
    """
    if any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
    ):
        raise ValueError("Invalid benchmark shape {!r}".format(shape))
    return tuple(shape)


def _call_plans(shape):
    for form, value in _NON_BLOCKING_FORMS:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"call": "non_blocking", "form": form, "value": value},
            builder_args=(shape, None, value),
        )


def _view_layouts(shape):
    """View branches that can deliver exactly ``shape``.

    A 1-D tensor of length L has no non-contiguous same-length view inside its
    own storage, so the view family starts at rank 2; the rank-1 scales stay
    covered by the call family.
    """
    if len(shape) < 2:
        return ()
    layouts = ["stride_slice", "transpose"]
    if len(shape) == 4:
        layouts.append("channels_last")
    return tuple(layouts)


def _view_plans(shape):
    for layout in _view_layouts(shape):
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"call": "view", "layout": layout},
            builder_args=(shape, layout, None),
        )


def _build_call_input(shape, value, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    if value is None:
        return (inp,)
    return (inp, {"non_blocking": value})


def _build_view_input(shape, layout, dtype, device):
    if layout == "transpose":
        # Swapping the first two extents makes the delivered tensor exactly the
        # listed shape while its strides stay transposed.
        parent = (shape[1], shape[0]) + tuple(shape[2:])
        return (utils.generate_tensor_input(parent, dtype, device).transpose(0, 1),)
    if layout == "channels_last":
        return (
            utils.generate_tensor_input(shape, dtype, device).to(
                memory_format=torch.channels_last
            ),
        )
    # Slicing the trailing extent out of a buffer whose rows are one element
    # wider keeps the delivered extents equal to the listed ones while the rows
    # are no longer contiguous.
    parent = tuple(shape[:-1]) + (shape[-1] + 1,)
    return (utils.generate_tensor_input(parent, dtype, device)[..., : shape[-1]],)


class CastCharBenchmark(base.UnaryPointwiseBenchmark):
    """The operator's call forms over the unary pointwise shape scales."""

    def get_case_iter(self, dtype):
        ordinal = 0
        for raw_shape in self.shapes:
            shape = _validated_shape(raw_shape)
            for plan in _call_plans(shape):
                yield self._case_from_plan(dtype, ordinal, plan)
                ordinal += 1

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, _, value = plan.builder_args
        return _build_call_input(shape, value, case.dtype, self.device)


class CastCharViewBenchmark(base.UnaryPointwiseBenchmark):
    """The schema-default call over transposed, channels-last and strided views."""

    def set_more_shapes(self):
        return super().set_more_shapes() + [
            _validated_shape(shape) for shape in _VIEW_EXTRA_SHAPES
        ]

    def get_case_iter(self, dtype):
        ordinal = 0
        for raw_shape in self.shapes:
            shape = _validated_shape(raw_shape)
            for plan in _view_plans(shape):
                yield self._case_from_plan(dtype, ordinal, plan)
                ordinal += 1

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, layout, _ = plan.builder_args
        return _build_view_input(shape, layout, case.dtype, self.device)


@pytest.mark.cast_Char
def test__cast_Char():
    bench = CastCharBenchmark(
        op_name="_cast_Char",
        torch_op=torch.ops.aten._cast_Char,
        dtypes=_benchmark_dtypes(),
        gems_op=getattr(flag_gems, "_cast_Char", None),
    )
    bench.run()


@pytest.mark.cast_Char
def test__cast_Char_views():
    bench = CastCharViewBenchmark(
        op_name="_cast_Char",
        torch_op=torch.ops.aten._cast_Char,
        dtypes=_benchmark_dtypes(),
        gems_op=getattr(flag_gems, "_cast_Char", None),
    )
    bench.run()
