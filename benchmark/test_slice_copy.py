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


def _input_fn(shape, dtype, device):
    # Slice along dim 1 over the middle half of the dimension with step 2; this
    # exercises the general (non-contiguous, non-inner1) kernel path.
    dim = 1
    dim_size = shape[dim]
    start = dim_size // 4
    end = dim_size - dim_size // 4
    step = 2
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, dim, start, end, step


def _get_gbps(bench_fn_args, latency):
    inp = bench_fn_args[0]
    # slice_copy reads the input and writes a fresh output of the sliced size.
    out_elems = inp.numel() // inp.size(bench_fn_args[1])
    out_size = out_elems * inp.element_size()
    io_amount = inp.numel() * inp.element_size() + out_size
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
        dim = 1
        dim_size = shape[dim]
        start = dim_size // 4
        end = dim_size - dim_size // 4
        step = 2
        inp = torch.randn(shape, dtype=dtype, device=device)
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
