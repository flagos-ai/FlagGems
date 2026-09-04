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

from functools import partial

import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base

CONTIGUOUS_SUFFIX_CASES = [
    ((1024, 4), 0),
    ((1, 2048, 8), 1),
    ((2, 8, 2048, 16), 2),
    ((2, 8, 2048, 32), 2),
    ((2, 8, 2048, 72), 2),
    ((1024, 64), 0),
]


def unpack_index_add_case(case):
    if isinstance(case[0], (list, tuple)):
        return tuple(case[0]), case[1]
    shape = tuple(case)
    return shape, 0 if len(shape) == 1 else 1


class TensorSelectBenchmark(base.GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # Speed Up Benchmark Test, Big Shape Will Cause Timeout
        if flag_gems.vendor_name == "kunlunxin":
            return []

        shapes = super().set_more_shapes()
        shapes = [
            # this filter is for scatter
            shape
            for shape in shapes
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]
        return shapes + CONTIGUOUS_SUFFIX_CASES


def index_add_gbps(bench_fn_args, latency):
    inp = bench_fn_args[0]
    index = bench_fn_args[2]
    src = bench_fn_args[3]
    io_amount = sum(
        [shape_utils.size_in_bytes(item) for item in [inp, inp, index, src]]
    )

    return io_amount * 1e-9 / (latency * 1e-3)


def index_add_input_fn(case, dtype, device):
    shape, dim = unpack_index_add_case(case)
    inp = torch.randn(shape, dtype=dtype, device=device)
    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max // 2 if index_max >= 2 else 1
    index = torch.randperm(index_len, device=device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=device)
    yield inp, dim, index, src


@pytest.mark.index_add
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_add():
    bench = TensorSelectBenchmark(
        op_name="index_add",
        torch_op=torch.index_add,
        input_fn=index_add_input_fn,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
        get_gbps=index_add_gbps,
    )
    bench.run()


def index_add__input_fn(case, dtype, device):
    shape, dim = unpack_index_add_case(case)
    inp = torch.randn(shape, dtype=dtype, device=device)
    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max // 2 if index_max >= 2 else 1
    index = torch.randperm(index_len, device=device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=device)
    yield inp, dim, index, src


CONTIGUOUS_SUFFIX_CONTENTION_CASES = [
    ((2, 8, 2048, 512), 2),  # wide contiguous suffix, tile path
    ((1024, 64), 0),  # dim-0 flat path
]
CONTENTION_DUP_FACTORS = [2, 8, 32, 128]


class IndexAddContentionBenchmark(TensorSelectBenchmark):
    def init_user_config(self):
        super().init_user_config()
        # This focused experiment must not inherit the default benchmark shapes.
        self.shapes = CONTIGUOUS_SUFFIX_CONTENTION_CASES


def index_add_contention_input_fn(case, dtype, device, dup_factor):
    shape, dim = unpack_index_add_case(case)
    inp = torch.randn(shape, dtype=dtype, device=device)
    index_max = shape[dim]
    index_len = index_max // 2 if index_max >= 2 else 1
    receiver_range = max(index_len // dup_factor, 1)
    index = torch.arange(index_len, device=device) % receiver_range
    src_shape = list(shape)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=device)
    yield inp, dim, index, src


@pytest.mark.parametrize(
    "op_name, torch_op",
    [
        pytest.param(
            "index_add", torch.index_add, marks=pytest.mark.index_add, id="functional"
        ),
        pytest.param(
            "index_add_",
            torch.Tensor.index_add_,
            marks=pytest.mark.index_add_,
            id="inplace",
        ),
    ],
)
def test_index_add_contention(op_name, torch_op):
    # The default input_fn draws a permutation, so atomics never contend.
    # Sweep receiver reuse factors to cover increasingly contended atomics.
    # Rows for one shape repeat in dup-factor order 2, 8, 32, 128.
    for dup_factor in CONTENTION_DUP_FACTORS:
        print(
            f"\n=== {op_name} contention tier: " f"receivers repeat ~{dup_factor}x ==="
        )
        bench = IndexAddContentionBenchmark(
            op_name=op_name,
            torch_op=torch_op,
            input_fn=partial(index_add_contention_input_fn, dup_factor=dup_factor),
            dtypes=[torch.float16, torch.bfloat16, torch.float32],
            get_gbps=index_add_gbps,
        )
        bench.run()


@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_add_():
    bench = TensorSelectBenchmark(
        op_name="index_add_",
        torch_op=torch.Tensor.index_add_,
        input_fn=index_add__input_fn,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
        get_gbps=index_add_gbps,
    )
    bench.run()
