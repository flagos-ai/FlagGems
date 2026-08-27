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

from . import base, consts

SEARCHSORTED_SHAPES = [
    ((1024,), (4096,), False, False, False),
    ((64, 256), (64, 256), False, True, False),
    ((256, 1024), (256, 512), False, False, True),
]

SEARCHSORTED_SCALAR_SHAPES = [
    ((1024,), False, False),
    ((4096,), True, False),
    ((8192,), False, True),
]

SEARCHSORTED_DTYPES = [
    torch.float16,
    torch.float32,
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "ascend",
            reason="Ascend native torch.searchsorted benchmark does not support bfloat16.",
        ),
    ),
    *consts.INT_DTYPES,
    torch.int8,
    torch.uint8,
]


def _bounds(dtype):
    if dtype == torch.uint8:
        return 0, 255
    if dtype == torch.int8:
        return -120, 120
    if dtype == torch.int16:
        return -2048, 2048
    return -4096, 4096


def _make_monotonic(shape, dtype, device):
    low, high = _bounds(dtype)
    base = torch.linspace(low, high, steps=shape[-1], dtype=torch.float32, device="cpu")
    if dtype.is_floating_point:
        base = base.to(dtype).to(device)
    else:
        base = base.round().to(dtype).to(device)
    view_shape = (1,) * (len(shape) - 1) + (shape[-1],)
    return base.reshape(view_shape).expand(shape).contiguous()


def _make_values(shape, dtype, device):
    low, high = _bounds(dtype)
    base = torch.linspace(low, high, steps=shape[-1], dtype=torch.float32, device="cpu")
    if dtype.is_floating_point:
        base = base.to(dtype).to(device)
    else:
        base = base.round().to(dtype).to(device)
    view_shape = (1,) * (len(shape) - 1) + (shape[-1],)
    return base.reshape(view_shape).expand(shape).contiguous()


def _make_scalar_value(dtype):
    return 0.0 if dtype.is_floating_point else 0


class SearchsortedBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "sorted_sequence shape, values shape, sorter, right, out_int32"

    def set_shapes(self, shape_file_path=None):
        self.shapes = SEARCHSORTED_SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for sequence_shape, values_shape, with_sorter, right, out_int32 in self.shapes:
            sorted_sequence = _make_monotonic(sequence_shape, dtype, self.device)
            sorter = None
            if with_sorter:
                sorted_sequence = torch.flip(sorted_sequence, dims=(-1,))
                sorter = torch.argsort(sorted_sequence, dim=-1)
            values = _make_values(values_shape, dtype, self.device)
            kwargs = {"out_int32": out_int32, "right": right}
            if sorter is not None:
                kwargs["sorter"] = sorter
            yield sorted_sequence, values, kwargs


class SearchsortedOutBenchmark(SearchsortedBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for sequence_shape, values_shape, with_sorter, right, out_int32 in self.shapes:
            sorted_sequence = _make_monotonic(sequence_shape, dtype, self.device)
            sorter = None
            if with_sorter:
                sorted_sequence = torch.flip(sorted_sequence, dims=(-1,))
                sorter = torch.argsort(sorted_sequence, dim=-1)
            values = _make_values(values_shape, dtype, self.device)
            out_dtype = torch.int32 if out_int32 else torch.int64
            out = torch.empty(values_shape, dtype=out_dtype, device=self.device)
            kwargs = {"out_int32": out_int32, "right": right, "out": out}
            if sorter is not None:
                kwargs["sorter"] = sorter
            yield sorted_sequence, values, kwargs


class SearchsortedScalarBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "sorted_sequence shape, right, out_int32"

    def set_shapes(self, shape_file_path=None):
        self.shapes = SEARCHSORTED_SCALAR_SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for sequence_shape, right, out_int32 in self.shapes:
            sorted_sequence = _make_monotonic(sequence_shape, dtype, self.device)
            value = _make_scalar_value(dtype)
            kwargs = {"out_int32": out_int32, "right": right}
            yield sorted_sequence, value, kwargs


class SearchsortedScalarOutBenchmark(SearchsortedScalarBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for sequence_shape, right, out_int32 in self.shapes:
            sorted_sequence = _make_monotonic(sequence_shape, dtype, self.device)
            value = _make_scalar_value(dtype)
            out_dtype = torch.int32 if out_int32 else torch.int64
            out = torch.empty((), dtype=out_dtype, device=self.device)
            kwargs = {"out_int32": out_int32, "right": right, "out": out}
            yield sorted_sequence, value, kwargs


@pytest.mark.searchsorted
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("dtype", SEARCHSORTED_DTYPES)
def test_searchsorted(dtype):
    bench = SearchsortedBenchmark(
        op_name="searchsorted",
        torch_op=torch.searchsorted,
        dtypes=[dtype],
    )
    bench.run()


@pytest.mark.searchsorted_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("dtype", SEARCHSORTED_DTYPES)
def test_searchsorted_scalar(dtype):
    bench = SearchsortedScalarBenchmark(
        op_name="searchsorted_scalar",
        torch_op=torch.searchsorted,
        dtypes=[dtype],
    )
    bench.run()


@pytest.mark.searchsorted_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("dtype", SEARCHSORTED_DTYPES)
def test_searchsorted_out(dtype):
    bench = SearchsortedOutBenchmark(
        op_name="searchsorted_out",
        torch_op=torch.searchsorted,
        dtypes=[dtype],
    )
    bench.run()


@pytest.mark.searchsorted_scalar_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("dtype", SEARCHSORTED_DTYPES)
def test_searchsorted_scalar_out(dtype):
    bench = SearchsortedScalarOutBenchmark(
        op_name="searchsorted_scalar_out",
        torch_op=torch.searchsorted,
        dtypes=[dtype],
    )
    bench.run()
