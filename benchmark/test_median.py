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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


class MedianNoDimBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape"
    with_out = False

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
            if self.with_out:
                out = torch.empty((), dtype=cur_dtype, device=self.device)
                yield inp, {"out": out}
            else:
                yield (inp,)


class MedianReductionBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape or [input shape, dim, keepdim]"
    with_out = False

    def get_input_iter(self, cur_dtype) -> Generator:
        for case_id, shape_spec in enumerate(self.shapes):
            if shape_spec and isinstance(shape_spec[0], (list, tuple)):
                shape = tuple(shape_spec[0])
                dim = int(shape_spec[1])
                keepdim = bool(shape_spec[2]) if len(shape_spec) > 2 else False
            else:
                shape = shape_spec
                keepdim = case_id % 3 == 0
                if len(shape) == 1:
                    dim = 0
                elif case_id % 2 == 0:
                    dim = len(shape) - 1
                else:
                    dim = 0
            inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
            if self.with_out:
                out_shape = list(shape)
                if keepdim:
                    out_shape[dim] = 1
                else:
                    del out_shape[dim]
                values = torch.empty(out_shape, dtype=cur_dtype, device=self.device)
                indices = torch.empty(out_shape, dtype=torch.int64, device=self.device)
                yield inp, dim, {"keepdim": keepdim, "out": (values, indices)}
            else:
                yield inp, dim, {"keepdim": keepdim}


@pytest.mark.median
def test_median():
    bench = MedianNoDimBenchmark(
        op_name="median",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.median_out
def test_median_out():
    bench = MedianNoDimBenchmark(
        op_name="median_out",
        torch_op=torch.ops.aten.median.out,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
        with_out=True,
    )
    bench.run()


@pytest.mark.skip_native(
    vendors=["enflame"],
    reason=(
        "aten::median.dim has no usable GCU kernel; the native call aborts the "
        "process (SIGABRT), so there is no native baseline to time"
    ),
)
@pytest.mark.median_dim
def test_median_dim():
    bench = MedianReductionBenchmark(
        op_name="median_dim",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.skip_native(
    vendors=["enflame"],
    reason=(
        "aten::median.dim_values has no usable GCU kernel; the native call "
        "aborts the process (SIGABRT), so there is no native baseline to time"
    ),
)
@pytest.mark.median_dim_values
def test_median_dim_values():
    bench = MedianReductionBenchmark(
        op_name="median_dim_values",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
        with_out=True,
    )
    bench.run()


def _entry_numel(entry):
    """Element count of a core-shape entry that may carry trailing arguments.

    `median_dim` shapes are either a bare shape (`[1536, 640]`) or
    `[shape, dim, keepdim]`; the backward VRAM filter needs the shape part only.
    Returns 0 for entries it cannot interpret so they are kept, not dropped.
    """
    shape = entry[0] if entry and isinstance(entry[0], (list, tuple)) else entry
    if not isinstance(shape, (list, tuple)):
        return 0
    numel = 1
    for dim in shape:
        if not isinstance(dim, int):
            return 0
        numel *= dim
    return numel


class _ForwardShapeBenchmark:
    """Resolve the core shapes of the forward operator for a backward bench.

    `Benchmark.__init__` appends `_backward` to `op_name`, but `core_shapes.yaml`
    only carries the forward entries (`median`, `median_dim`). Resolve the
    forward entry instead of silently falling back to `Benchmark` shapes so the
    backward benchmark covers exactly the same cases as the forward one.
    """

    def set_shapes(self, shape_file_path=None):
        op_name = self.op_name
        if op_name.endswith("_backward"):
            self.op_name = op_name[: -len("_backward")]
        try:
            if flag_gems.vendor_name == "enflame" and self.is_backward:
                # `Benchmark.set_shapes` applies its backward VRAM filter with
                # `math.prod()` over every core-shape entry. `median_dim` also
                # ships argument tuples such as `[[32, 384, 128], 1, false]`,
                # where `math.prod()` raises TypeError before any kernel runs.
                # Load without that filter, then apply it to the shape part of
                # each entry so the memory budget is still honoured.
                self.is_backward = False
                try:
                    super().set_shapes(shape_file_path)
                finally:
                    self.is_backward = True
                limit = self.MAX_BACKWARD_ELEMENTS
                if limit is not None:
                    kept = [s for s in self.shapes if _entry_numel(s) <= limit]
                    if kept:
                        self.shapes = kept
            else:
                super().set_shapes(shape_file_path)
        finally:
            self.op_name = op_name


class MedianNoDimBackwardBenchmark(_ForwardShapeBenchmark, MedianNoDimBenchmark):
    """Times `torch.autograd.grad` of the global median.

    `is_backward=True` makes `Benchmark.unpack_to_args_kwargs` turn the input
    into a differentiable leaf and `_benchmark_callable` time
    `torch.autograd.grad` with a retained graph.
    """


class MedianDimBackwardBenchmark(_ForwardShapeBenchmark, MedianReductionBenchmark):
    def _benchmark_callable(self, op, *args, **kwargs):
        inp = args[0]
        dim = args[1]
        keepdim = kwargs.get("keepdim", False)
        values = op(inp, dim=dim, keepdim=keepdim)[0]
        dout = torch.randn_like(values)
        xs = [arg for arg in args if torch.is_tensor(arg) and arg.requires_grad]
        return (
            lambda: torch.autograd.grad(
                (values,), xs, grad_outputs=(dout,), retain_graph=True
            ),
            xs,
        )


@pytest.mark.median_backward
def test_median_backward():
    bench = MedianNoDimBackwardBenchmark(
        op_name="median",
        torch_op=torch.median,
        gems_op=flag_gems.median,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()


@pytest.mark.skip_native(
    vendors=["enflame"],
    reason=(
        "aten::median.dim_values has no usable GCU kernel; the native call "
        "aborts the process (SIGABRT), so there is no native baseline to time"
    ),
)
@pytest.mark.median_backward
def test_median_dim_backward():
    bench = MedianDimBackwardBenchmark(
        op_name="median_dim",
        torch_op=torch.median,
        gems_op=flag_gems.median_dim,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
