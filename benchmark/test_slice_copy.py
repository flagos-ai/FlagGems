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


class SliceCopyBenchmark(base.GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # Speed Up Benchmark Test, Big Shape Will Cause Timeout.
        if flag_gems.vendor_name == "kunlunxin":
            return []
        # 2D shapes only; slice_copy slices along dim 1 with step 2, so keep the
        # second axis large enough to make the slice non-trivial.
        return [(10000, 2**i) for i in (8, 16)]


def _slice_params(shape, dim):
    """Slice the middle half of ``dim`` with step 2."""
    dim_size = shape[dim]
    start = dim_size // 4
    end = dim_size - dim_size // 4
    step = 2
    return start, end, step


# For a 2-D input, slicing dim 1 leaves inner == 1, so only the inner1 kernel
# runs. Slicing dim 0 gives inner == shape[1] > 1, which is the general
# (strided, inner > 1) kernel. Benchmark both so each path is measured.
_BENCH_DIMS = (1, 0)


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    for dim in _BENCH_DIMS:
        start, end, step = _slice_params(shape, dim)
        yield inp, dim, start, end, step


def _slice_numel(inp, dim, start, end, step):
    slice_len = max(0, (end - start + step - 1) // step)
    numel = slice_len
    for d, size in enumerate(inp.shape):
        if d != dim:
            numel *= size
    return numel


def _get_gbps(bench_fn_args, latency):
    # slice_copy touches output.numel() elements on each side: it gathers only
    # the selected slice out of the input and writes an equally sized output.
    # Counting the whole input as read overstates the traffic whenever the slice
    # is a strict subset of the dimension.
    inp, dim, start, end, step = bench_fn_args[:5]
    out_elems = _slice_numel(inp, dim, start, end, step)
    io_amount = 2 * out_elems * inp.element_size()
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.slice_copy
def test_slice_copy():
    bench = SliceCopyBenchmark(
        op_name="slice_copy",
        torch_op=torch.slice_copy,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )
    bench.run()


@pytest.mark.slice_copy_out
def test_slice_copy_out():
    def torch_op(inp, dim, start, end, step, out=None):
        return torch.ops.aten.slice_copy.Tensor_out(inp, dim, start, end, step, out=out)

    def _input_fn_out(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        for dim in _BENCH_DIMS:
            start, end, step = _slice_params(shape, dim)
            out_shape = list(shape)
            out_shape[dim] = max(0, (end - start + step - 1) // step)
            out = torch.empty(out_shape, dtype=dtype, device=device)
            yield inp, dim, start, end, step, {"out": out}

    bench = SliceCopyBenchmark(
        op_name="slice_copy_out",
        torch_op=torch_op,
        input_fn=_input_fn_out,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )
    bench.run()
