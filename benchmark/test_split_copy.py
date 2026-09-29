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

# Four performance-relevant scales. Every bare shape is timed on its leading and
# its trailing axis; a shape file may also request one explicit axis with a
# {"shape": [...], "split_size": n, "dim": d} entry.
SPLIT_COPY_SHAPES = [
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype != torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def _bad_extent(extent):
    return isinstance(extent, bool) or not isinstance(extent, int) or extent < 0


def _normalize_descriptor(entry):
    """Validate one requested shape and return (shape, split_size, dim)."""
    if isinstance(entry, dict):
        unknown = set(entry) - {"shape", "split_size", "dim"}
        if unknown:
            raise ValueError(
                f"unsupported split_copy descriptor keys: {sorted(unknown)}"
            )
        if "shape" not in entry:
            raise ValueError("split_copy descriptor needs a 'shape' key")
        shape = entry["shape"]
        split_size, dim = entry.get("split_size"), entry.get("dim")
    else:
        shape, split_size, dim = entry, None, None

    # Invalid requests are rejected here instead of being dropped silently, so a
    # malformed shape file cannot shrink the benchmark set unnoticed.
    if not isinstance(shape, (list, tuple)):
        raise ValueError(
            f"split_copy shape must be a sequence of extents, got {shape!r}"
        )
    if not shape:
        raise ValueError("split_copy requires a tensor of rank >= 1")
    if any(_bad_extent(extent) for extent in shape):
        raise ValueError(
            f"split_copy extents must be non-negative integers, got {shape!r}"
        )

    rank = len(shape)
    if dim is not None:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(f"split_copy dim must be an integer, got {dim!r}")
        if not -rank <= dim <= rank - 1:
            raise ValueError(f"split_copy dim {dim} is out of range for rank {rank}")
    if split_size is not None:
        if isinstance(split_size, bool) or not isinstance(split_size, int):
            raise ValueError(
                f"split_copy split_size must be an integer, got {split_size!r}"
            )
        if split_size < 0:
            raise ValueError(
                f"split_copy split_size must be non-negative, got {split_size}"
            )
        # split_size 0 is only valid when the split axis is itself empty (it then
        # yields a single full-size part), which is what the native op reports.
        if split_size == 0 and shape[0 if dim is None else dim] != 0:
            raise ValueError("split_copy split_size 0 needs an empty split axis")
    return tuple(shape), split_size, dim


def _plan(shape, split_size, dim):
    return base.BenchmarkCasePlan(
        shape={"input": list(shape), "dim": dim},
        params={"split_size": split_size, "dim": dim},
        builder_args=(shape, split_size, dim),
    )


def _plans(shape, split_size, dim):
    """Cases for one validated request; a bare shape is timed on two axes."""
    if split_size is None and dim is None:
        yield _plan(shape, max(1, shape[-1] // 4), len(shape) - 1)
        if len(shape) > 1:
            yield _plan(shape, max(1, shape[0] // 4), 0)
        return
    axis = 0 if dim is None else dim
    if split_size is None:
        split_size = max(1, shape[axis] // 4)
    yield _plan(shape, split_size, axis)


def _case_fn(entry, dtype):
    del dtype
    yield from _plans(*_normalize_descriptor(entry))


def _build_inputs_fn(plan, dtype, device):
    shape, split_size, dim = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {"split_size": split_size, "dim": dim}


class SplitCopyBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over the operator's shape set."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SPLIT_COPY_SHAPES)


@pytest.mark.split_copy
def test_split_copy():
    bench = SplitCopyBenchmark(
        op_name="split_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.split_copy,
        gems_op=getattr(flag_gems, "split_copy", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
