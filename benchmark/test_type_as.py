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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# ``type_as`` is a pure memory-bound cast: ``self`` is streamed while ``other``
# supplies both the target dtype and the target device. Both operands are built
# on the selected benchmark device, so a case measures one device-local cast.
# The generic default shape list starts at 1024**3 elements and this operator
# has no ``core_shapes.yaml`` entry, so the suite declares cast-sized shapes of
# its own. One shape list serves every benchmark level, so listing, execution
# and ``--case-id`` replay always agree; a ``type_as`` (or
# ``TypeAsBenchmark``) entry in a ``--shape_file`` replaces it through
# ``OperatorBenchmark.set_shapes``.
#
# A shape entry may name the cast endpoints instead of only a shape:
#
#     type_as:
#       shapes:
#         - [1024, 1024]
#         - {shape: [512, 512], source_dtype: int32, target_dtype: float32}
#
# A plain shape keeps the default pairing below. A mapping that names
# ``source_dtype`` is a fully explicit pair and is planned exactly once, even
# when that source is not one of ``BENCH_DTYPES``; a mapping that names only
# ``target_dtype`` keeps the default source for each benchmarked dtype. Every
# endpoint is checked against the static capability flags, and an endpoint the
# backend does not support is left out of the plan rather than replaced.
TYPE_AS_SHAPES = [
    (1024,),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64, 60),
    (8192, 1024),
    (8, 64, 4096),
    (512, 128, 64),
]

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_DTYPES_BY_NAME = {}
for _dtype in (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    *_FP8_DTYPES,
):
    _DTYPES_BY_NAME[str(_dtype)] = _dtype
    _DTYPES_BY_NAME[str(_dtype).replace("torch.", "")] = _dtype


def _endpoint_supported(dtype):
    """Static capability report for a cast endpoint; nothing is probed here."""
    device = flag_gems.runtime.device
    if dtype == torch.float64:
        return device.support_fp64
    if dtype == torch.bfloat16:
        return device.support_bf16
    if dtype == torch.int64:
        return device.support_int64
    if dtype in _FP8_DTYPES:
        return device.support_fp8
    return True


BENCH_DTYPES = [dtype for dtype in consts.FLOAT_DTYPES if _endpoint_supported(dtype)]

_OTHER_DTYPE = {
    torch.float16: torch.float32,
    torch.float32: torch.float16,
    torch.bfloat16: torch.float32,
}


def _other_dtype(dtype):
    """A target dtype distinct from ``dtype`` so the cast never short-circuits."""
    if dtype in _OTHER_DTYPE:
        return _OTHER_DTYPE[dtype]
    return torch.float32 if dtype is not torch.float32 else torch.float16


def _validated_shape(shape):
    """Reject metadata that cannot describe a tensor, before any allocation."""
    if not isinstance(shape, (tuple, list)):
        raise TypeError(f"shape must be a tuple or list, got {type(shape).__name__}")
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise TypeError(f"shape extents must be ints, got {extent!r}")
        if extent < 0:
            raise ValueError(f"shape extents must be non-negative, got {extent!r}")
    return tuple(shape)


def _dtype_from_name(value, field):
    if isinstance(value, torch.dtype):
        return value
    if not isinstance(value, str) or value not in _DTYPES_BY_NAME:
        raise TypeError(
            f"{field} must be a torch dtype or a dtype name such as 'int32', "
            f"got {value!r}"
        )
    return _DTYPES_BY_NAME[value]


def _parse_shape_entry(entry):
    """Split a shape entry into (shape, requested source dtype, target dtype)."""
    if isinstance(entry, dict):
        unknown = set(entry) - {"shape", "source_dtype", "target_dtype"}
        if unknown:
            raise TypeError(f"unknown shape entry keys: {sorted(unknown)}")
        if "shape" not in entry:
            raise TypeError("a shape entry mapping must carry a 'shape' key")
        source = entry.get("source_dtype")
        target = entry.get("target_dtype")
        return (
            _validated_shape(entry["shape"]),
            None if source is None else _dtype_from_name(source, "source_dtype"),
            None if target is None else _dtype_from_name(target, "target_dtype"),
        )
    return _validated_shape(entry), None, None


def _generate(shape, dtype, device):
    """Materialize ``shape`` as ``dtype``.

    ``utils.generate_tensor_input`` only covers the declared float, int16/32,
    bool and complex lists and silently yields ``None`` for anything else, so
    the remaining cast endpoints (int8, uint8, int64, float64, float8) get a
    generation of their own instead of being dropped.
    """
    tensor = utils.generate_tensor_input(shape, dtype, device)
    if tensor is not None:
        return tensor
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    info = torch.iinfo(dtype)
    # A full integer range is only accepted as a CPU bound pair; move it after.
    return torch.randint(info.min, info.max, shape, dtype=dtype).to(device)


def _build_inputs_fn(plan, dtype, device):
    del dtype  # the plan carries the real source/target pair
    shape, source_dtype, target_dtype = plan.builder_args
    inp = _generate(shape, source_dtype, device)
    # Only the dtype and device of the second operand matter; one element keeps
    # it tiny.
    other = torch.zeros((1,), dtype=target_dtype, device=device)
    return inp, other


class TypeAsBenchmark(OperatorBenchmark):
    """``OperatorBenchmark`` with this operator's own default shapes and pairs."""

    def __init__(self, *args, **kwargs):
        # The framework hands ``case_fn`` only the current dtype pass, so the
        # benchmark installs its own planner, which also knows the benchmarked
        # dtypes.
        kwargs.setdefault("case_fn", self.plan_cases)
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=TYPE_AS_SHAPES)

    def plan_cases(self, shape_entry, dtype):
        """Plan the cases of one dtype pass, including pinned cast pairs."""
        shape, requested_source, requested_target = _parse_shape_entry(shape_entry)
        if requested_source is None:
            source = dtype
            target = (
                _other_dtype(source) if requested_target is None else requested_target
            )
        else:
            # An explicit pair is a property of the case, not of the framework's
            # dtype pass: whichever pass is the designated bucket emits it once,
            # with the requested source in both the metadata and the builder,
            # and the other passes emit nothing. ``base.py`` only schedules cases
            # whose dtype is benchmarked, so the bucket ownership decides how
            # often the case runs, and it must not depend on the requested
            # source appearing in the default float list.
            if dtype != self.to_bench_dtypes[0]:
                return
            source = requested_source
            target = (
                _other_dtype(source) if requested_target is None else requested_target
            )
        if not _endpoint_supported(source) or not _endpoint_supported(target):
            return
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"source_dtype": str(source), "target_dtype": str(target)},
            builder_args=(shape, source, target),
        )


@pytest.mark.type_as
def test_type_as():
    bench = TypeAsBenchmark(
        op_name="type_as",
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.type_as,
        gems_op=getattr(flag_gems, "type_as", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
