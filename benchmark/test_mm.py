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


def mm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.t()
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mm():
    bench = base.BlasBenchmark(
        op_name="mm",
        input_fn=mm_input_fn,
        torch_op=torch.Tensor.mm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


class MmSelfTransposeBenchmark(base.GenericBenchmark2DOnly):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        return []

    def get_tflops(self, op, *args, **kwargs):
        m, k = args[0].shape
        return 2 * m * m * k


def _input_fn(shape, cur_dtype, device):
    m, k = shape
    inp = torch.randn([k, m], dtype=cur_dtype, device=device).t()

    yield inp,


def torch_mm_self_transpose(inp):
    return torch.mm(inp, inp.t())


@pytest.mark.mm_self_transpose
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mm_self_transpose_benchmark():
    bench = MmSelfTransposeBenchmark(
        op_name="mm_self_transpose",
        input_fn=_input_fn,
        torch_op=torch_mm_self_transpose,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def mm_out_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        out = torch.empty([m, n], dtype=cur_dtype, device=device)
        yield inp1, inp2.t(), {"out": out}
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        out = torch.empty([m, n], dtype=cur_dtype, device=device)
        yield inp1, inp2, {"out": out}


@pytest.mark.mm_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mm_out():
    bench = base.BlasBenchmark(
        op_name="mm_out",
        input_fn=mm_out_input_fn,
        torch_op=torch.mm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


# Public shape metadata from scientific inference workloads; no model data.
HYGON_SCIENTIFIC_MM_SHAPES = [
    (303, 320, 320),
    (303, 33, 320),
    (29403, 1, 120),
    (1220, 128, 128),
    (2806, 128, 128),
    (1220, 256, 128),
    (1220, 128, 256),
    (2806, 256, 128),
    (2806, 128, 256),
    (4819, 128, 128),
    (126636, 128, 128),
    (4819, 256, 128),
    (4819, 128, 256),
    (126636, 256, 128),
    (126636, 128, 256),
]


class HygonScientificMmBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = HYGON_SCIENTIFIC_MM_SHAPES

    def get_input_iter(self, dtype):
        for m, n, k in self.shapes:
            a = torch.randn((m, k), dtype=dtype, device=self.device)
            b = torch.randn((n, k), dtype=dtype, device=self.device).t()
            yield a, b

    def get_tflops(self, op, *args, **kwargs):
        m, k = args[0].shape
        n = args[1].shape[1]
        return 2 * m * n * k


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon scientific matrix schedules"
)
def test_hygon_scientific_mm():
    bench = HygonScientificMmBenchmark(
        op_name="mm", torch_op=torch.mm, dtypes=[torch.float32]
    )
    bench.run()
