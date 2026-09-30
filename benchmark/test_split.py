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
from .generated_operator_utils import OperatorBenchmark

# One benchmark covers every overload that reaches the public name:
#   split.Tensor(t, SymInt split_size, int dim=0) -> Tensor[]
#   split.sizes(t, SymInt[] sizes, int dim=0)     -> Tensor[]
#   split.str(str self, str? separator, int max)  -> str[]

# The str overload takes no tensor, so a string workload is described by its
# arguments instead of a shape. Each descriptor is a 4-tuple that starts with
# "str"; the remaining entries are the (self, separator, max) arguments.
_LONG_FIELDS = ",".join(str(i) for i in range(1024))
SPLIT_STRINGS = [
    ("str", "a,b,c", None, -1),
    ("str", _LONG_FIELDS, ",", -1),
    ("str", _LONG_FIELDS, ",", 1),
    ("str", "  a  b c  ", None, -1),
]

# int8/uint8/int64/float8 are absent from consts.FLOAT_DTYPES and friends, so the
# grid is assembled from all of the shared dtype groups.
BENCH_DTYPES = (
    consts.FLOAT_DTYPES
    + consts.EXTRA_INT_DTYPES
    + consts.INT_DTYPES
    + consts.BOOL_DTYPES
    + consts.COMPLEX_DTYPES
    + [dtype for dtype in consts.FP8_DTYPES if dtype is not None]
)


def _chunk(extent):
    return max(1, extent // 3)


def _partition(extent, parts):
    size, rest = divmod(extent, parts)
    sizes = [size] * parts
    for i in range(rest):
        sizes[i] += 1
    return sizes


def _is_string_case(value):
    return isinstance(value, (tuple, list)) and len(value) == 4 and value[0] == "str"


def _is_rank_supported(value):
    # aten::split rejects a 0-dim input ("split expects at least a
    # 1-dimensional tensor"), so () is the only shape that is dropped from
    # whatever set the caller or core_shapes.yaml supplies.
    return _is_string_case(value) or len(value) >= 1


def _build_inputs_fn(plan, dtype, device):
    if _is_string_case(plan.builder_args):
        _, self_value, separator, max_value = plan.builder_args
        # The trailing dict is merged into kwargs by unpack_to_args_kwargs; an
        # explicit None separator is the schema default and stays an argument.
        return self_value, {"separator": separator, "max": max_value}

    shape, form, split_arg, dim = plan.builder_args
    # split only rearranges metadata, so no part exposes the payload: an
    # uninitialized allocation keeps the timed work on the metadata itself.
    inp = torch.empty(shape, dtype=dtype, device=device)
    if dim is None:
        return inp, split_arg
    return inp, split_arg, dim


class SplitBenchmark(OperatorBenchmark):
    def __init__(self, *args, **kwargs):
        # case_fn is bound to the instance because it has to know which dtype
        # list is being benchmarked.
        super().__init__(*args, case_fn=self._case_fn, **kwargs)

    def _case_fn(self, shape, dtype):
        if _is_string_case(shape):
            # The str overload carries no dtype, so its plans are emitted once,
            # under the first benchmark dtype, instead of once per dtype.
            if dtype is not self.to_bench_dtypes[0]:
                return
            _, self_value, separator, max_value = shape
            yield base.BenchmarkCasePlan(
                shape={"input": self_value},
                params={"form": "str", "separator": separator, "max": max_value},
                builder_args=shape,
            )
            return

        last = len(shape) - 1
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={
                "form": "split_size",
                "dim": "default",
                "split_size": _chunk(shape[0]),
            },
            builder_args=(shape, "split_size", _chunk(shape[0]), None),
        )
        if last > 0:
            yield base.BenchmarkCasePlan(
                shape={"input": list(shape)},
                params={
                    "form": "split_size",
                    "dim": last,
                    "split_size": _chunk(shape[last]),
                },
                builder_args=(shape, "split_size", _chunk(shape[last]), last),
            )
        sizes = _partition(shape[0], 3)
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"form": "sizes", "dim": 0, "sizes": list(sizes)},
            builder_args=(shape, "sizes", list(sizes), 0),
        )

    def set_shapes(self, shape_file_path=None):
        # Keep the shared shape set (core_shapes.yaml stays in charge, and the
        # shared comprehensive extras keep applying) and append the str
        # workloads, which a shape-based description cannot express.
        super().set_shapes(shape_file_path)
        shapes = [shape for shape in self.shapes if _is_rank_supported(shape)]
        self.shapes = list(
            dict.fromkeys(tuple(shape) for shape in shapes + list(SPLIT_STRINGS))
        )


@pytest.mark.split
def test_split():
    bench = SplitBenchmark(
        op_name="split",
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.split,
        gems_op=getattr(flag_gems, "split", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
