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

from . import accuracy_utils as utils

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads" or not hasattr(flag_gems, "mm_w8a8_fp8"),
    reason="MThreads FP8 backend",
)


def _mm_w8a8_fp8_reference(a, b):
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)

    a_fp32 = a.float()
    a_scale = a_fp32.abs().amax(dim=1).clamp_min(1e-10) / fp8_info.max
    a_fp8 = (a_fp32 / a_scale[:, None]).clamp(fp8_info.min, fp8_info.max).to(fp8_dtype)

    b_fp32 = b.float()
    b_scale = b_fp32.abs().amax(dim=0).clamp_min(1e-10) / fp8_info.max
    b_fp8 = (b_fp32 / b_scale[None, :]).clamp(fp8_info.min, fp8_info.max).to(fp8_dtype)

    return torch.mm(a_fp8.float(), b_fp8.float()) * a_scale[:, None] * b_scale[None, :]


@pytest.mark.mm_w8a8_fp8
@pytest.mark.parametrize(
    "M, N, K",
    [
        (1, 16, 16),
        (2, 32, 32),
        (8, 64, 64),
        (16, 128, 64),
        (32, 128, 128),
        (64, 256, 128),
        (128, 256, 256),
        (192, 512, 512),
        (256, 768, 1024),
        (512, 1024, 1024),
    ],
)
def test_mm_w8a8_fp8(M, N, K):
    dtype = torch.bfloat16
    torch.manual_seed(0)
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_out = utils.to_reference(_mm_w8a8_fp8_reference(mat1, mat2), True)

    res_out = flag_gems.mm_w8a8_fp8(mat1, mat2, out_dtype=dtype)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    res_out_reused = flag_gems.mm_w8a8_fp8_out(mat1, mat2, out=out)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
    utils.gems_assert_close(res_out_reused, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.parametrize(
    "M,N,K",
    [
        (1, 1, 1),
        (7, 33, 15),
        (16, 128, 64),
        (31, 65, 129),
        (32, 64, 7168),
        (64, 256, 7168),
        (128, 256, 4096),
        (256, 768, 1024),
        (512, 1024, 2048),
        (64, 64, 32768),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("column_major", [True, False])
def test_mm_w8a8_fp8_mthreads_prequantized(M, N, K, dtype, column_major):
    torch.manual_seed(42)
    a = torch.randn((M, K), device=flag_gems.device).to(dtype)
    b = torch.randn((N, K) if column_major else (K, N), device=flag_gems.device).to(
        dtype
    )
    if column_major:
        b = b.T
    ref = a.float() @ b.float()
    out_storage = torch.empty((M, N * 2), device=a.device, dtype=torch.bfloat16)
    out = out_storage[:, ::2]
    result = flag_gems.mm_w8a8_fp8_out(a, b, out=out)
    assert result is out
    torch.testing.assert_close(result, ref.to(out.dtype), rtol=0.016, atol=0.01)
    result_fp32 = flag_gems.mm_w8a8_fp8(a, b, out_dtype=torch.float32)
    torch.testing.assert_close(result_fp32, ref, rtol=5e-4, atol=0.003)


@pytest.mark.mm_w8a8_fp8
def test_mm_w8a8_fp8_mthreads_graph_updates():
    a = torch.randn((32, 128), device=flag_gems.device, dtype=torch.bfloat16)
    b = torch.randn((128, 64), device=flag_gems.device, dtype=torch.bfloat16)
    out = torch.empty((32, 64), device=a.device, dtype=a.dtype)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(3):
            flag_gems.mm_w8a8_fp8_out(a, b, out=out)
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph):
            flag_gems.mm_w8a8_fp8_out(a, b, out=out)
    torch.musa.current_stream().wait_stream(stream)
    for _ in range(2):
        a.normal_()
        b.normal_()
        graph.replay()
        ref = _mm_w8a8_fp8_reference(a, b).to(out.dtype)
        torch.testing.assert_close(out, ref, rtol=0.016, atol=0.01)


@pytest.mark.mm_w8a8_fp8
def test_mm_w8a8_fp8_mthreads_empty():
    for m, n, k in [(0, 32, 16), (32, 0, 16), (32, 16, 0)]:
        a = torch.empty((m, k), device=flag_gems.device, dtype=torch.float8_e4m3fn)
        b = torch.empty((k, n), device=flag_gems.device, dtype=a.dtype)
        result = flag_gems.mm_w8a8_fp8(a, b)
        torch.testing.assert_close(
            result, torch.zeros((m, n), device=a.device, dtype=torch.bfloat16)
        )


@pytest.mark.mm_w8a8_fp8
@pytest.mark.parametrize(
    "out_dtype", [torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_mm_w8a8_fp8_mthreads_output_dtype(out_dtype):
    a = torch.randn((64, 32), device=flag_gems.device).to(torch.float8_e4m3fn).T
    b = torch.randn((64, 32), device=flag_gems.device).to(a.dtype)
    ref = a.float() @ b.float()
    result = flag_gems.mm_w8a8_fp8(a, b, out_dtype=out_dtype)
    assert result.dtype == out_dtype
    torch.testing.assert_close(
        result.float(), ref.to(out_dtype).float(), rtol=0.001, atol=0.002
    )


@pytest.mark.mm_w8a8_fp8
def test_mm_w8a8_fp8_mthreads_fp16_and_invalid_out():
    a = torch.randn((64, 32), device=flag_gems.device, dtype=torch.float16).T
    b = torch.randn((64, 32), device=flag_gems.device, dtype=torch.float16)
    ref = _mm_w8a8_fp8_reference(a, b).to(a.dtype)
    first = flag_gems.mm_w8a8_fp8(a, b)
    second = flag_gems.mm_w8a8_fp8(a, b)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, ref, rtol=0.002, atol=0.002)
    wrong = torch.empty((32, 33), device=a.device, dtype=a.dtype)
    with pytest.raises(ValueError, match="out has an incompatible shape"):
        flag_gems.mm_w8a8_fp8_out(a, b, out=wrong)


@pytest.mark.mm_w8a8_fp8
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mm_w8a8_fp8_mthreads_broadcast(dtype):
    a = torch.randn((1, 64), device=flag_gems.device).to(dtype).expand(32, 64)
    b = torch.randn((64, 1), device=flag_gems.device).to(dtype).expand(64, 32)
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    result = flag_gems.mm_w8a8_fp8(a, b)
    torch.testing.assert_close(result, ref, rtol=0.016, atol=0.01)
