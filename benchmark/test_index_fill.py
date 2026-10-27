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

from . import base, consts

INDEX_RATIOS = ("1/16", "1/2", "full")
INDEX_FILL_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
MIN_SELECTED_NUMEL = 16 * 1024
DIM0_INDEX_LENGTHS = {
    (4096, 256): (8, 256, 2048),
    (4096, 4096): (512, 2048, 4096),
    (8192, 4096): (512, 4096, 8192),
}


class IndexFillBenchmark(base.GenericBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS
    DEFAULT_SHAPES = [
        (65536,),
        (4096, 256),
        (4096, 4096),
    ]
    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        self.shapes = list(self.DEFAULT_SHAPES)
        if (
            base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
            and not base.Config.query
        ):
            self.shapes = list(dict.fromkeys(self.shapes + self.set_more_shapes()))

    def set_more_shapes(self):
        return [
            (8192, 4096),
            (200, 40999, 3),
        ]

    def _clone_inplace_args(self, args):
        if not self.is_inplace:
            return args
        return (args[0].clone(), *args[1:])

    def get_latency(self, op, *args, **kwargs):
        if base.Config.mode == consts.BenchMode.OPERATOR:
            # Keep one-time Triton loading out of the adaptive iteration count.
            op(*self._clone_inplace_args(args), **kwargs)
            base.torch_device_fn.synchronize()
        return super().get_latency(op, *self._clone_inplace_args(args), **kwargs)


def _generate_input(shape, dtype, device):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.randint(-10, 10, shape, dtype=dtype, device=device)


def _dims_for_shape(shape):
    return (0,) if len(shape) == 1 else (0, 1)


def _index_len(dim_size, ratio):
    if ratio == "1/16":
        return max(1, dim_size // 16)
    if ratio == "1/2":
        return max(1, dim_size // 2)
    if ratio == "full":
        return dim_size
    raise ValueError(f"Unknown index ratio: {ratio}")


def _make_index(dim_size, index_len, device):
    return torch.randperm(dim_size, device=device)[:index_len]


def _scalar_value(dtype):
    if dtype == torch.bool:
        return True
    if dtype.is_floating_point:
        return 3.14159
    return 3


def _base_inputs(shape, dtype, device):
    for dim in _dims_for_shape(shape):
        dim_size = shape[dim]
        index_lengths = DIM0_INDEX_LENGTHS.get(tuple(shape), ()) if dim == 0 else ()
        if not index_lengths:
            index_lengths = tuple(_index_len(dim_size, ratio) for ratio in INDEX_RATIOS)
        for index_len in dict.fromkeys(index_lengths):
            selected_numel = math.prod(shape) // dim_size * index_len
            if selected_numel < MIN_SELECTED_NUMEL and dim != 0:
                continue
            inp = _generate_input(shape, dtype, device)
            index = _make_index(dim_size, index_len, device)
            yield inp, dim, index


def index_fill_input_fn(shape, dtype, device):
    for inp, dim, index in _base_inputs(shape, dtype, device):
        yield inp, dim, index, _scalar_value(dtype)


@pytest.mark.index_fill
def test_index_fill():
    bench = IndexFillBenchmark(
        op_name="index_fill",
        input_fn=index_fill_input_fn,
        torch_op=torch.index_fill,
        dtypes=INDEX_FILL_DTYPES,
    )
    bench.run()


@pytest.mark.index_fill_
def test_index_fill_():
    bench = IndexFillBenchmark(
        op_name="index_fill_",
        input_fn=index_fill_input_fn,
        torch_op=torch.Tensor.index_fill_,
        dtypes=INDEX_FILL_DTYPES,
    )
    bench.run()
