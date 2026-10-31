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

"""Regression coverage for MetaX trajectory-derived GEMM/GLU paths."""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "metax", reason="MetaX kernels"
)


@pytest.fixture(autouse=True)
def restore_tf32():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(17)
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(33, 65, 17), (64, 64, 71), (65, 127, 33)])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.mm
def test_metax_mm_tails(dtype, shape, transposed):
    m, n, k = shape
    a = torch.randn((m, k), device=flag_gems.device, dtype=dtype)
    b = torch.randn((n, k) if transposed else (k, n), device=a.device, dtype=dtype)
    if transposed:
        b = b.t()
    expected = a.double() @ b.double()
    actual = flag_gems.mm(a, b)
    torch.testing.assert_close(actual.double(), expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("operator", ["mm", "bmm"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.mm
@pytest.mark.bmm
def test_metax_zero_reduction(operator, dtype):
    batch = () if operator == "mm" else (2,)
    a = torch.empty((*batch, 33, 0), device=flag_gems.device, dtype=dtype)
    b = torch.empty((*batch, 0, 65), device=a.device, dtype=dtype)
    actual = getattr(flag_gems, operator)(a, b)
    assert actual.shape == (*batch, 33, 65)
    assert torch.count_nonzero(actual).item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["transposed", "broadcast", "sliced"])
@pytest.mark.bmm
def test_metax_bmm_strides(dtype, layout):
    a = torch.randn((2, 33, 35), device=flag_gems.device, dtype=dtype)
    b = torch.randn((2, 65, 35), device=a.device, dtype=dtype).transpose(1, 2)
    if layout == "broadcast":
        a = a[:1].expand(2, -1, -1)
    elif layout == "sliced":
        a = torch.randn((2, 66, 35), device=a.device, dtype=dtype)[:, ::2, :]
    actual = flag_gems.bmm(a, b)
    expected = torch.bmm(a.double(), b.double())
    torch.testing.assert_close(actual.double(), expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("operator", ["mm", "bmm"])
@pytest.mark.mm
@pytest.mark.bmm
def test_metax_fp32_policy(operator):
    from flag_gems.runtime.backend._metax.ops._tuned_gemm_policy import use_tuned_gemm

    shape = (64, 512) if operator == "mm" else (2, 64, 512)
    a = torch.full(shape, 1.0003, device=flag_gems.device)
    b = a.transpose(-1, -2).contiguous()
    assert not use_tuned_gemm(a, b)
    expected = a.double() @ b.double()
    actual = getattr(flag_gems, operator)(a, b)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-4)
    torch.backends.cuda.matmul.allow_tf32 = True
    assert use_tuned_gemm(a, b)
    actual = getattr(flag_gems, operator)(a, b)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "case", ["fast", "ragged", "transpose", "slice", "empty", "axis"]
)
@pytest.mark.glu
def test_metax_glu_layouts(dtype, case):
    x = torch.randn((5, 8, 64), device=flag_gems.device, dtype=dtype)
    dim = -1
    if case == "ragged":
        x = x[..., :62]
    elif case == "transpose":
        x = x.transpose(0, 1)
    elif case == "slice":
        x = x[..., ::2]
    elif case == "empty":
        x = x[:0]
    elif case == "axis":
        dim = 1
    actual = flag_gems.glu(x, dim)
    expected = torch.nn.functional.glu(x.double(), dim)
    tol = {torch.float16: 0.002, torch.bfloat16: 0.01, torch.float32: 2e-6}[dtype]
    torch.testing.assert_close(actual.double(), expected, rtol=tol, atol=tol)
