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

import importlib

import pytest
import torch
import triton

import flag_gems

from .accuracy_utils import gems_assert_close, gems_assert_equal

requires_int8 = pytest.mark.skipif(
    "grouped_gemm_w8a8" not in vars(flag_gems),
    reason="requires a registered grouped INT8 GEMM implementation",
)


def make_grouped_gemm_case(
    lengths, n, k, device, seed=20261019, *, dtype=torch.int8, column_major_scales=False
):
    torch.manual_seed(seed)
    m, e = sum(lengths), len(lengths)
    if dtype == torch.int8:
        A = torch.randint(-128, 128, (m, k), dtype=dtype, device=device)
        B = torch.randint(-128, 128, (e, n, k), dtype=dtype, device=device)
        As = 0.005 + 0.01 * torch.rand((m, 1), device=device)
        Bs = 0.005 + 0.01 * torch.rand((e, n, 1), device=device)
        C = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    else:
        A = torch.randn((m, k), device=device).to(dtype)
        B = torch.randn((e, n, k), device=device).to(dtype)
        if column_major_scales:
            As = (0.01 + 0.02 * torch.rand((k // 128, m), device=device)).T
        else:
            As = 0.01 + 0.02 * torch.rand((m, k // 128), device=device)
        Bs = 0.01 + 0.02 * torch.rand((e, triton.cdiv(n, 128), k // 128), device=device)
        C = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    rows = torch.tensor(lengths, dtype=torch.int32, device=device)
    indices = torch.repeat_interleave(
        torch.arange(e, dtype=torch.int32, device=device), rows, output_size=m
    )
    return A, As, B, Bs, C, indices, rows


def deepgemm_grouped_gemm(A, As, B, Bs, C, m_indices, m_rows):
    import deep_gemm

    deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_nopad(
        (A, As), (B, Bs), C, m_indices, m_rows
    )


def grouped_gemm_cpu_reference(A, As, B, Bs, lengths):
    A, As, B, Bs = (t.cpu() for t in (A, As, B, Bs))
    n, k = B.shape[1:]
    output = torch.empty((A.shape[0], n), dtype=torch.bfloat16)
    start = 0
    for expert, count in enumerate(lengths):
        end = start + count
        if A.dtype == torch.int8:
            dot = A[start:end].double() @ B[expert].double().T
            output[start:end] = (
                dot.float() * As[start:end] * Bs[expert, :, 0]
            ).bfloat16()
        else:
            acc = torch.zeros((count, n), dtype=torch.float64)
            for block in range(k // 128):
                section = slice(block * 128, (block + 1) * 128)
                dot = A[start:end, section].double() @ B[expert, :, section].double().T
                bs = Bs[expert, :, block].double().repeat_interleave(128)[:n]
                acc += dot * As[start:end, block, None].double() * bs[None, :]
            output[start:end] = acc.bfloat16()
        start = end
    return output


@pytest.mark.parametrize(
    "lengths,n,k",
    [
        ([0, 1, 3, 0, 7], 65, 129),
        ([1, 32, 0, 65], 128, 256),
        ([0, 257, 3], 64, 128),
        ([0, 257, 3], 257, 128),
        ([0, 129, 3], 65, 129),
        ([0, 129, 3], 17, 0),
        ([0, 129, 3], 0, 128),
        ([1] + [0] * 255 + [13], 64, 6144),
        ([0, 1, 0], 17, 1),
        ([0, 0, 0], 64, 128),
    ],
)
@requires_int8
@pytest.mark.grouped_gemm_w8a8
def test_grouped_gemm_w8a8(lengths, n, k):
    args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
    expected = grouped_gemm_cpu_reference(*args[:4], lengths)
    flag_gems.grouped_gemm_w8a8(*args)
    gems_assert_equal(args[4].cpu(), expected)


@pytest.mark.parametrize("n,k", [(512, 6144), (6144, 256)])
@requires_int8
@pytest.mark.grouped_gemm_w8a8
def test_grouped_gemm_w8a8_deepgemm(n, k):
    pytest.importorskip("deep_gemm")
    lengths = [1] * 32 + [0] * 224 + [13]
    args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
    expected = torch.empty_like(args[4])
    deepgemm_grouped_gemm(*args[:4], expected, *args[5:])
    flag_gems.grouped_gemm_w8a8(*args)
    gems_assert_close(args[4], expected, torch.bfloat16, atol=1e-3)


@pytest.mark.parametrize("n", [64, 65])
@requires_int8
@pytest.mark.grouped_gemm_w8a8
def test_grouped_gemm_w8a8_int8_extremes(n):
    args = make_grouped_gemm_case([1, 0, 1], n, 6144, flag_gems.device)
    args[0].fill_(-128)
    args[2][0].fill_(-128)
    args[2][2].fill_(127)
    expected = grouped_gemm_cpu_reference(*args[:4], [1, 0, 1])
    flag_gems.grouped_gemm_w8a8(*args)
    gems_assert_equal(args[4].cpu(), expected)


@pytest.mark.parametrize(
    "lengths",
    [[0, 1, 17, 0, 35] + [9] * 59, [0, 566] + [0] * 61 + [18], [0, 257, 0, 511]],
)
@pytest.mark.parametrize("values", ["random", "extreme", "cancellation"])
@requires_int8
@pytest.mark.grouped_gemm_w8a8
def test_grouped_gemm_w8a8_pipeline(lengths, values):
    n, k = 68, 6144
    args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
    if values != "random":
        args[0].fill_(-128)
        args[2].fill_(-128)
        if values == "cancellation":
            args[0].fill_(127)
            args[2].fill_(127)
            args[0][:, k // 2 :] = -127
            args[0][:, k // 3] = 126
    expected = grouped_gemm_cpu_reference(*args[:4], lengths)
    flag_gems.grouped_gemm_w8a8(*args)
    gems_assert_equal(args[4].cpu(), expected)


@requires_int8
@pytest.mark.grouped_gemm_w8a8
def test_grouped_gemm_w8a8_changing_expert_rows():
    for lengths, n, k in (
        ([0, 257, 0, 511], 64, 128),
        ([256, 0, 512, 0], 128, 256),
        ([256, 0, 512, 0], 64, 128),
        ([0, 257, 0, 511], 128, 256),
        ([2] * 256, 68, 128),
        ([0, 511] + [0] * 253 + [1], 68, 128),
        ([0, 3, 0, 11] + [0] * 12, 68, 128),
        ([7, 0, 7, 0] + [0] * 12, 68, 128),
        ([0, 257, 0, 511], 68, 128),
        ([4] * 128, 32, 4096),
        ([0, 511] + [0] * 125 + [1], 32, 4096),
    ):
        args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
        expected = grouped_gemm_cpu_reference(*args[:4], lengths)
        flag_gems.grouped_gemm_w8a8(*args)
        gems_assert_equal(args[4].cpu(), expected)


@pytest.mark.parametrize(
    "async_a,async_b,block_m,block_k,swap,stages",
    [(True, False, 64, 128, False, 3), (False, True, 128, 64, True, 4)],
)
@pytest.mark.parametrize("values", ["random", "extreme", "cancellation"])
@requires_int8
@pytest.mark.grouped_gemm_w8a8
def test_grouped_gemm_w8a8_async(
    async_a, async_b, block_m, block_k, swap, stages, values
):
    module = importlib.import_module(flag_gems.grouped_gemm_w8a8.__module__)
    lengths = [0, 17, 65, 0, 131, 1]
    n, k = 128, 256 if values == "random" else 6144
    args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
    if values != "random":
        args[0].fill_(-128)
        args[2].fill_(-128)
        if values == "cancellation":
            # Preserve an odd residual after partial sums exceed FP32's exact range.
            args[0].fill_(127)
            args[2].fill_(127)
            args[0][:, k // 2 :] = -127
            args[0][:, k // 3] = 126
    expected = grouped_gemm_cpu_reference(*args[:4], lengths)
    m, e = sum(lengths), len(lengths)
    workspace = (
        torch.empty(m + e * (block_m - 1), dtype=torch.int32, device=args[0].device),
        torch.empty(
            triton.cdiv(m, block_m) + e, dtype=torch.int32, device=args[0].device
        ),
        torch.empty(1, dtype=torch.int32, device=args[0].device),
    )
    module.grouped_gemm_metadata[(e,)](
        *args[5:],
        *workspace,
        m,
        e,
        BLOCK_M=block_m,
        BLOCK_E=triton.next_power_of_2(e),
        num_warps=4,
    )
    tiles_m = triton.cdiv(triton.cdiv(m, block_m) + e - 1, 8) * 8
    module.grouped_gemm_tiled_kernel[(tiles_m,)](
        *args,
        *workspace,
        m,
        n,
        k,
        e,
        triton.next_power_of_2(e),
        BLOCK_M=block_m,
        BLOCK_N=128,
        BLOCK_K=block_k,
        WARPS=4,
        STAGES=stages,
        ASYNC_A=async_a,
        ASYNC_B=async_b,
        SCALAR_PROBE=True,
        SWAP=swap,
        TLE_PIPELINE=True,
        NAIVE=False,
        GROUP_M=8,
        ADAPTIVE_M=True,
        num_warps=4,
        num_stages=stages,
    )
    gems_assert_equal(args[4].cpu(), expected)


@pytest.mark.parametrize("cache,evict", [(".cg", ""), (".ca", "evict_last")])
@requires_int8
@pytest.mark.grouped_gemm_w8a8
def test_grouped_gemm_w8a8_gemv_cache_tail(cache, evict):
    module = importlib.import_module(flag_gems.grouped_gemm_w8a8.__module__)
    lengths = [1, 0, 8, 3]
    n, k = 65, 260
    args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
    expected = grouped_gemm_cpu_reference(*args[:4], lengths)
    module.grouped_gemv_int8_kernel[(triton.cdiv(n, 8), sum(lengths))](
        *args[:6],
        sum(lengths),
        n,
        k,
        len(lengths),
        BLOCK_N=8,
        BLOCK_K4=32,
        WARPS=1,
        STAGES=1,
        TLE_PIPELINE=False,
        CACHE_B=cache,
        EVICT_B=evict,
        num_warps=1,
        num_stages=1,
    )
    gems_assert_equal(args[4].cpu(), expected)


requires_fp8 = pytest.mark.skipif(
    "grouped_gemm_w8a8_fp8" not in vars(flag_gems),
    reason="requires a registered unpadded grouped FP8 GEMM backend",
)


def make_grouped_fp8_case(lengths, n, k, device, column_major_scales=False):
    return make_grouped_gemm_case(
        lengths,
        n,
        k,
        device,
        seed=42,
        dtype=torch.float8_e4m3fn,
        column_major_scales=column_major_scales,
    )


def mate_grouped_fp8(a, sa, b, sb, out, m_indices, m_rows):
    from mate.gemm import ragged_m_moe_gemm_8bit

    ragged_m_moe_gemm_8bit(
        (a, sa), (b, sb), m_rows, out, gemm_mode="per_expert", backend="mubin"
    )


@requires_fp8
@pytest.mark.grouped_gemm_w8a8_fp8
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize(
    "lengths,n,k",
    [
        ([1, 0, 7], 64, 128),
        ([63, 65, 0, 129], 192, 256),
        ([0, 0], 128, 128),
        ([7, 3], 256, 6144),
        ([1, 3], 6144, 128),
        ([1, 129], 512, 1024),
        ([1] * 8 + [0] * 24, 256, 6144),
        ([1, 273, 512], 256, 6144),
        ([1, 273, 512], 512, 128),
        ([1, 0, 7], 192, 384),
        ([1, 63, 65], 192, 128),
        ([0, 193, 0, 1351], 768, 128),
        ([0, 17, 137, 401], 512, 768),
    ],
)
def test_grouped_gemm_w8a8_fp8(lengths, n, k, column_major):
    args = make_grouped_fp8_case(lengths, n, k, flag_gems.device, column_major)
    expected = grouped_gemm_cpu_reference(*args[:4], lengths)
    # Leave real rows directly adjacent to BF16 sentinels on both sides.
    backing = torch.full(
        (sum(lengths) + 2, n), 7.0, device=flag_gems.device, dtype=torch.bfloat16
    )
    out = backing[1:-1]
    out.fill_(float("nan"))
    assert flag_gems.grouped_gemm_w8a8_fp8(*args[:4], out, *args[5:]) is None
    gems_assert_close(out.cpu(), expected, torch.bfloat16)
    guards = backing.cpu()[[0, -1]]
    gems_assert_equal(guards, torch.full_like(guards, 7.0))


@requires_fp8
@pytest.mark.grouped_gemm_w8a8_fp8
@pytest.mark.parametrize("n,k", [(256, 6144), (6144, 128)])
def test_grouped_gemm_w8a8_fp8_native(n, k):
    pytest.importorskip("mate.gemm")
    args = make_grouped_fp8_case([1, 0, 17, 129], n, k, flag_gems.device)
    expected = args[4].clone()
    mate_grouped_fp8(*args[:4], expected, *args[5:])
    flag_gems.grouped_gemm_w8a8_fp8(*args)
    gems_assert_close(args[4], expected, torch.bfloat16)


@requires_fp8
@pytest.mark.grouped_gemm_w8a8_fp8
@pytest.mark.parametrize("n,k", [(256, 512), (512, 128)])
def test_grouped_gemm_w8a8_fp8_changing_routes(n, k):
    args = make_grouped_fp8_case([257, 0, 513], n, k, flag_gems.device)
    for lengths in ([257, 0, 513], [0, 769, 1], [770, 0, 0], [257, 0, 513]):
        args[6].copy_(torch.tensor(lengths, device=args[6].device, dtype=torch.int32))
        args[5].copy_(
            torch.repeat_interleave(
                torch.arange(3, dtype=torch.int32, device=args[5].device),
                args[6],
                output_size=770,
            )
        )
        args[4].fill_(float("nan"))
        expected = grouped_gemm_cpu_reference(*args[:4], lengths)
        flag_gems.grouped_gemm_w8a8_fp8(*args)
        gems_assert_close(args[4].cpu(), expected, torch.bfloat16)
