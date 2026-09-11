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


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=dtype, device=device)
        yield bias, inp1, inp2.t(),
    else:
        inp2 = torch.randn([k, n], dtype=dtype, device=device)
        yield bias, inp1, inp2,


class AddmmVectorBiasBenchmark(base.BlasBenchmark):
    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for b, m, n, k in self.shapes:
            yield from self.input_fn(b, m, n, k, dtype, self.device, True)

    def get_tflops(self, op, *args, **kwargs):
        _, mat1, mat2 = args
        return mat1.shape[0] * mat2.shape[1] * (2 * mat1.shape[1] + 1)

    def record_shapes(self, bias, mat1, mat2, **kwargs):
        return {
            "bias": bias.size(),
            "mat1": mat1.size(),
            "mat2": mat2.size(),
            "mat2_layout": ("column-major" if mat2.stride(0) == 1 else "row-major"),
        }


def _input_fn_vector_bias(b, m, n, k, dtype, device, b_column_major):
    mat1 = torch.randn((m, k), dtype=dtype, device=device)
    if b_column_major:
        mat2 = torch.randn((n, k), dtype=dtype, device=device).t()
    else:
        mat2 = torch.randn((k, n), dtype=dtype, device=device)
    bias = torch.randn((n,), dtype=dtype, device=device)
    yield bias, mat1, mat2


def _input_fn_vector_bias_out(b, m, n, k, dtype, device, b_column_major):
    for bias, mat1, mat2 in _input_fn_vector_bias(
        b, m, n, k, dtype, device, b_column_major
    ):
        out = torch.empty((m, n), dtype=dtype, device=device)
        yield bias, mat1, mat2, {"out": out}


@pytest.mark.addmm
def test_addmm(monkeypatch):
    bench = base.BlasBenchmark(
        op_name="addmm",
        input_fn=_input_fn,
        torch_op=torch.addmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


@pytest.mark.addmm
def test_addmm_vector_bias(monkeypatch):
    bench = AddmmVectorBiasBenchmark(
        op_name="addmm_vector_bias",
        input_fn=_input_fn_vector_bias,
        torch_op=torch.addmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_dtype(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=torch.float32, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=dtype, device=device)
        yield bias, inp1, inp2.t(), torch.float32
    else:
        inp2 = torch.randn([k, n], dtype=dtype, device=device)
        yield bias, inp1, inp2, torch.float32


@pytest.mark.addmm_dtype
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.8"),
    reason="The operator addmm.dtype was added starting from 2.8.0",
)
@pytest.mark.skipif(
    flag_gems.vendor_name in ("ascend", "mthreads"),
    reason="Issue #5385: Native torch.addmm benchmark does not support out_dtype.",
)
def test_addmm_dtype(monkeypatch):
    bench = base.BlasBenchmark(
        op_name="addmm_dtype",
        input_fn=_input_fn_dtype,
        torch_op=torch.ops.aten.addmm.dtype,
        dtypes=consts.FP16_BF16_DTYPES,
    )

    bench.run()


def _input_fn_dtype_out(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=torch.float32, device=device)
    out = torch.empty([m, n], dtype=torch.float32, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=dtype, device=device)
        yield bias, inp1, inp2.t(), torch.float32, out
    else:
        inp2 = torch.randn([k, n], dtype=dtype, device=device)
        yield bias, inp1, inp2, torch.float32, out


@pytest.mark.addmm_dtype_out
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.8"),
    reason="The operator addmm.dtype_out was added starting from 2.8.0",
)
@pytest.mark.skipif(
    flag_gems.vendor_name in ("ascend", "mthreads"),
    reason="Issue #5385: Native torch.addmm benchmark does not support out_dtype.",
)
def test_addmm_dtype_out(monkeypatch):
    bench = base.BlasBenchmark(
        op_name="addmm_dtype_out",
        input_fn=_input_fn_dtype_out,
        torch_op=lambda bias, mat1, mat2, out_dtype, out: torch.ops.aten.addmm.dtype_out(
            bias, mat1, mat2, out_dtype, beta=1.0, alpha=1.0, out=out
        ),
        dtypes=consts.FP16_BF16_DTYPES,
    )

    bench.run()


def _input_fn_out(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=dtype, device=device)
    out = torch.empty([m, n], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=dtype, device=device)
        yield bias, inp1, inp2.t(), {"out": out}
    else:
        inp2 = torch.randn([k, n], dtype=dtype, device=device)
        yield bias, inp1, inp2, {"out": out}


@pytest.mark.addmm_out
def test_addmm_out(monkeypatch):
    bench = base.BlasBenchmark(
        op_name="addmm_out",
        input_fn=_input_fn_out,
        torch_op=torch.addmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


@pytest.mark.addmm_out
def test_addmm_out_vector_bias(monkeypatch):
    bench = AddmmVectorBiasBenchmark(
        op_name="addmm_out_vector_bias",
        input_fn=_input_fn_vector_bias_out,
        torch_op=torch.addmm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
