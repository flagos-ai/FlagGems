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


def make_grouped_gemm_case(
    lengths,
    n,
    k,
    device,
    seed=20261019,
    *,
    dtype=torch.int8,
    block_scaled=False,
    strided=False,
    out=None,
):
    torch.manual_seed(seed)
    m, e = sum(lengths), len(lengths)
    if block_scaled:
        A = torch.randn((m, k), device=device).to(dtype)
        B = torch.randn((e, n, k), device=device).to(dtype)
        As = 0.01 + 0.02 * torch.rand((m, k // 128), device=device)
        Bs = 0.01 + 0.02 * torch.rand((e, (n + 127) // 128, k // 128), device=device)
    else:
        A = torch.randint(-128, 128, (m, k), dtype=torch.int8, device=device)
        B = torch.randint(-128, 128, (e, n, k), dtype=torch.int8, device=device)
        As = 0.005 + 0.01 * torch.rand((m, 1), device=device)
        Bs = 0.005 + 0.01 * torch.rand((e, n, 1), device=device)
        if dtype != torch.int8:
            A = (A.cpu().float() / 64).to(dtype).to(device)
            B = (B.cpu().float() / 64).to(dtype).to(device)
    if strided:
        As = As.T.contiguous().T
        Bs = Bs.transpose(1, 2).contiguous().transpose(1, 2)
    C = torch.empty((m, n), dtype=torch.bfloat16, device=device) if out is None else out
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


def npu_grouped_gemm(A, As, B, Bs, C, m_indices, m_rows):
    import torch_npu

    accumulator = torch_npu.npu_grouped_matmul(
        [A],
        [B.transpose(-1, -2)],
        group_list=m_rows.to(torch.int64),
        group_list_type=1,
        split_item=3,
        group_type=0,
        output_dtype=torch.int32,
    )[0]
    C.copy_(accumulator.float() * As * Bs[m_indices.to(torch.int64), :, 0])


def get_grouped_gemm_int8_reference():
    if flag_gems.vendor_name == "ascend":
        pytest.importorskip("torch_npu")
        return npu_grouped_gemm
    pytest.importorskip("deep_gemm")
    return deepgemm_grouped_gemm


def grouped_gemm_cpu_reference(A, As, B, Bs, lengths, *, block_scaled=False):
    A, As, B, Bs = (t.cpu() for t in (A, As, B, Bs))
    if block_scaled:
        A, As, B, Bs = (t.double() for t in (A, As, B, Bs))
    output = torch.empty((A.shape[0], B.shape[1]), dtype=torch.bfloat16)
    start = 0
    for expert, count in enumerate(lengths):
        end = start + count
        if block_scaled:
            acc = torch.zeros((count, B.shape[1]), dtype=torch.float64)
            for block in range(A.shape[1] // 128):
                section = slice(block * 128, (block + 1) * 128)
                dot = A[start:end, section] @ B[expert, :, section].T
                scale_b = Bs[expert, :, block].repeat_interleave(128)[: B.shape[1]]
                acc += dot * As[start:end, block, None] * scale_b[None, :]
            output[start:end] = acc.bfloat16()
        else:
            dot = A[start:end].double() @ B[expert].double().T
            output[start:end] = (
                dot.float() * As[start:end] * Bs[expert, :, 0]
            ).bfloat16()
        start = end
    return output


def assert_grouped_gemm_result(
    args,
    lengths,
    *,
    block_scaled=False,
    storage=None,
    scale_factor=1.0,
    normalize=False,
    relative=False,
):
    expected = grouped_gemm_cpu_reference(*args[:4], lengths, block_scaled=block_scaled)
    actual = args[4].cpu()
    if block_scaled:
        if normalize:
            # Preserve the scale-independent check for cancellation with order-one scales.
            rms = expected.float().square().mean().sqrt()
            torch.testing.assert_close(
                actual.float() / rms, expected.float() / rms, rtol=0.016, atol=1e-3
            )
        else:
            gems_assert_close(
                actual,
                expected,
                torch.bfloat16,
                reduce_dim=args[0].shape[1] if scale_factor > 1 else 1,
            )
        if relative and expected.numel():
            error = (actual.float() - expected.float()).norm() / expected.float().norm()
            assert error < 1e-3
    elif args[0].dtype == torch.int8:
        gems_assert_equal(actual, expected)
    else:
        # Keep the existing per-channel FP8 tolerance: one BF16 rounding step.
        torch.testing.assert_close(
            actual, expected, rtol=torch.finfo(torch.bfloat16).eps, atol=0
        )
    if storage is not None:
        assert torch.all(storage[0] == 17) and torch.all(storage[-1] == 17)


BLOCK_SCALED_FP8 = "grouped_gemm_w8a8_fp8" in vars(flag_gems)


@pytest.fixture
def grouped_gemm_operator():
    if BLOCK_SCALED_FP8:
        return flag_gems.grouped_gemm_w8a8_fp8
    if "grouped_gemm_w8a8" in vars(flag_gems):
        return flag_gems.grouped_gemm_w8a8
    pytest.skip("requires a registered grouped GEMM implementation")


@pytest.fixture
def grouped_gemm_inputs():
    def _make(
        lengths, n, k, *, dtype=None, strided=False, unaligned="", scale_factor=1.0
    ):
        if dtype is None:
            dtype = torch.float8_e4m3fn if BLOCK_SCALED_FP8 else torch.int8
        m = sum(lengths)
        storage, out = None, None
        if BLOCK_SCALED_FP8:
            unaligned_out = unaligned in ("out", "all")
            shape = (m * n + 2,) if unaligned_out else (m + 2, n)
            storage = torch.full(
                shape, 17, device=flag_gems.device, dtype=torch.bfloat16
            )
            out = storage[1:-1].view(m, n)
        args = list(
            make_grouped_gemm_case(
                lengths,
                n,
                k,
                flag_gems.device,
                seed=42 if BLOCK_SCALED_FP8 else 20261019,
                dtype=dtype,
                block_scaled=BLOCK_SCALED_FP8,
                strided=strided,
                out=out,
            )
        )
        if BLOCK_SCALED_FP8:
            args[1].mul_(scale_factor)
            args[3].mul_(scale_factor)
        for name, index in (("a", 0), ("b", 2)):
            if unaligned in (name, "ab", "all"):
                original = args[index]
                backing = torch.empty(
                    original.numel() + 1, dtype=original.dtype, device=original.device
                )
                args[index] = backing[1:].view(original.shape)
                args[index].copy_(original)
        args[4].fill_(float("nan"))
        return args, storage

    return _make


@pytest.mark.grouped_gemm_w8a8
@pytest.mark.parametrize("strided", [False, True] if BLOCK_SCALED_FP8 else [False])
@pytest.mark.parametrize("scale_factor", [1.0, 100.0] if BLOCK_SCALED_FP8 else [1.0])
@pytest.mark.parametrize(
    "lengths,n,k",
    (
        [
            ([0, 0, 0], 64, 128),
            ([0, 1, 3, 0, 7], 192, 384),
            ([1] + [0] * 255 + [13], 64, 6144),
            ([1, 32, 0, 65], 128, 256),
            ([0, 257, 3], 64, 128),
            ([0, 257, 0, 511], 192, 384),
            ([0, 566] + [0] * 61 + [18], 64, 6144),
        ]
        if BLOCK_SCALED_FP8
        else (
            [
                ([0, 1, 3, 0, 7], 65, 129),
                ([1, 32, 0, 65], 128, 256),
                ([0, 257, 3], 64, 128),
                ([1] + [0] * 255 + [13], 64, 6144),
                ([0, 129, 0, 3968], 257, 2051),
                ([0, 129] + [0] * 254 + [4000], 257, 2051),
                ([0, 1, 0], 17, 1),
                ([0, 0, 0], 64, 128),
            ]
            if flag_gems.vendor_name == "metax"
            else [
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
            ]
        )
    ),
)
def test_grouped_gemm_w8a8(
    grouped_gemm_operator, grouped_gemm_inputs, lengths, n, k, strided, scale_factor
):
    args, storage = grouped_gemm_inputs(
        lengths, n, k, strided=strided, scale_factor=scale_factor
    )
    pointer = args[4].data_ptr()
    assert grouped_gemm_operator(*args) is None
    assert args[4].data_ptr() == pointer
    assert_grouped_gemm_result(
        args,
        lengths,
        block_scaled=BLOCK_SCALED_FP8,
        storage=storage,
        scale_factor=scale_factor,
        relative=BLOCK_SCALED_FP8,
    )


@pytest.mark.grouped_gemm_w8a8
@pytest.mark.skipif(
    not BLOCK_SCALED_FP8 and flag_gems.vendor_name != "metax",
    reason="requires a registered FP8 grouped GEMM backend",
)
@pytest.mark.parametrize(
    "dtype",
    (
        [torch.float8_e4m3fn]
        if BLOCK_SCALED_FP8
        else [torch.int8, torch.float8_e4m3fn, torch.float8_e4m3fnuz]
    ),
)
@pytest.mark.parametrize(
    "lengths,n,k,strided,unaligned,scale_factor,normalize",
    (
        (
            [
                (lengths, 192, 384, True, unaligned, 1.0, False)
                for lengths in ([1, 33, 0, 29], [0, 257, 0, 511])
                for unaligned in ("a", "b", "out", "all")
            ]
            + [
                (lengths, 192, k, strided, "ab" if unaligned else "", 100.0, True)
                for lengths in ([257, 1, 767, 0], [0, 1025, 0, 0], [511, 1, 1537, 0])
                for k in (128, 384)
                for strided in (False, True)
                for unaligned in (False, True)
            ]
            + [([257, 1, 767, 0], 192, 128, True, "out", 1.0, False)]
        )
        if BLOCK_SCALED_FP8
        else [
            (*case, False, "", 1.0, False)
            for case in [
                ([0, 3, 1], 17, 0),
                ([0, 3, 1], 0, 16),
                ([0, 0], 16, 16),
                ([0, 17, 65, 0, 131, 1], 65, 129),
                ([0, 257, 3], 128, 256),
                ([0, 513, 0], 256, 256),
            ]
        ]
    ),
)
def test_grouped_gemm_w8a8_dtypes_tails(
    grouped_gemm_operator,
    grouped_gemm_inputs,
    lengths,
    n,
    k,
    dtype,
    strided,
    unaligned,
    scale_factor,
    normalize,
):
    args, storage = grouped_gemm_inputs(
        lengths,
        n,
        k,
        dtype=dtype,
        strided=strided,
        unaligned=unaligned,
        scale_factor=scale_factor,
    )
    assert grouped_gemm_operator(*args) is None
    assert_grouped_gemm_result(
        args,
        lengths,
        block_scaled=BLOCK_SCALED_FP8,
        storage=storage,
        scale_factor=scale_factor,
        normalize=normalize,
        relative=normalize,
    )


@pytest.mark.grouped_gemm_w8a8
@pytest.mark.skipif(
    not BLOCK_SCALED_FP8 and flag_gems.vendor_name != "metax",
    reason="requires a registered FP8 grouped GEMM backend",
)
@pytest.mark.parametrize(
    "m,k",
    (
        [(m, k) for m in (16, 512, 2048) for k in (128, 384)]
        if BLOCK_SCALED_FP8
        else [(None, None)]
    ),
)
def test_grouped_gemm_w8a8_changing_expert_rows(
    grouped_gemm_operator, grouped_gemm_inputs, m, k
):
    storage = None
    if BLOCK_SCALED_FP8:
        lengths, n = [m // 4] * 4, 192
        args, storage = grouped_gemm_inputs(lengths, n, k, strided=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            grouped_gemm_operator(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            grouped_gemm_operator(*args)
        cases = [
            (lengths, n, k)
            for lengths in ([0, m - 1, 0, 1], [m // 2, 0, m // 2, 0], [0, 0, 0, m])
        ]
    else:
        cases = [
            ([0, 257, 0, 511], 64, 128),
            ([256, 0, 512, 0], 128, 256),
            ([256, 0, 512, 0], 64, 128),
            ([0, 257, 0, 511], 128, 256),
        ]
    for lengths, n, k in cases:
        if BLOCK_SCALED_FP8:
            args[6].copy_(
                torch.tensor(lengths, device=args[6].device, dtype=torch.int32)
            )
            args[5].copy_(
                torch.repeat_interleave(
                    torch.arange(4, device=args[5].device, dtype=torch.int32),
                    args[6],
                    output_size=m,
                )
            )
            args[4].fill_(float("nan"))
            graph.replay()
        else:
            args, storage = grouped_gemm_inputs(lengths, n, k)
            grouped_gemm_operator(*args)
        assert_grouped_gemm_result(
            args, lengths, block_scaled=BLOCK_SCALED_FP8, storage=storage
        )


@pytest.mark.skipif(
    flag_gems.vendor_name not in ("ascend", "enflame", "thead")
    or "grouped_gemm_w8a8" not in vars(flag_gems),
    reason="requires a registered Ascend or PPU grouped GEMM backend",
)
class TestGroupedInt8GEMM:

    @pytest.mark.parametrize("n,k", [(512, 6144), (6144, 256)])
    @pytest.mark.skipif(
        flag_gems.vendor_name == "enflame",
        reason=(
            "enflame backend has no deep_gemm build, so the grouped GEMM "
            "reference cannot be produced"
        ),
    )
    @pytest.mark.grouped_gemm_w8a8
    def test_grouped_gemm_w8a8_native(self, n, k):
        lengths = [1] * 32 + [0] * 224 + [13]
        args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
        expected = torch.empty_like(args[4])
        get_grouped_gemm_int8_reference()(*args[:4], expected, *args[5:])
        flag_gems.grouped_gemm_w8a8(*args)
        gems_assert_close(args[4].cpu(), expected.cpu(), torch.bfloat16, atol=1e-3)

    @pytest.mark.parametrize("n", [64, 65])
    @pytest.mark.grouped_gemm_w8a8
    def test_grouped_gemm_w8a8_int8_extremes(self, n):
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
    @pytest.mark.grouped_gemm_w8a8
    def test_grouped_gemm_w8a8_pipeline(self, lengths, values):
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

    @pytest.mark.grouped_gemm_w8a8
    def test_grouped_gemm_w8a8_changing_expert_rows(self):
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
    @pytest.mark.skipif(
        flag_gems.vendor_name == "enflame",
        reason=(
            "enflame backend ships a single tiled kernel and does not expose "
            "the thead grouped_gemm_metadata / grouped_gemm_tiled_kernel "
            "workspace ABI this case drives directly"
        ),
    )
    @pytest.mark.grouped_gemm_w8a8
    @pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="PPU kernel options")
    def test_grouped_gemm_w8a8_async(
        self, async_a, async_b, block_m, block_k, swap, stages, values
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
            torch.empty(
                m + e * (block_m - 1), dtype=torch.int32, device=args[0].device
            ),
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
    @pytest.mark.skipif(
        flag_gems.vendor_name == "enflame",
        reason=(
            "enflame backend has no grouped_gemv_int8_kernel with the thead "
            "cache-hint constexpr ABI this case drives directly"
        ),
    )
    @pytest.mark.grouped_gemm_w8a8
    @pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="PPU cache options")
    def test_grouped_gemm_w8a8_gemv_cache_tail(self, cache, evict):
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


@pytest.mark.skipif(
    flag_gems.vendor_name != "metax" or "grouped_gemm_w8a8" not in vars(flag_gems),
    reason="requires the registered metax grouped GEMM backend",
)
class TestMetaXGroupedGEMM:

    @pytest.mark.grouped_gemm_w8a8
    def test_grouped_gemm_w8a8_int8_extremes(self):
        args = make_grouped_gemm_case([1, 0, 1], 65, 6144, flag_gems.device)
        args[0].fill_(-128)
        args[2][0].fill_(-128)
        args[2][2].fill_(127)
        expected = grouped_gemm_cpu_reference(*args[:4], [1, 0, 1])
        pointer = args[4].data_ptr()
        assert flag_gems.grouped_gemm_w8a8(*args) is None
        assert args[4].data_ptr() == pointer
        gems_assert_equal(args[4].cpu(), expected)

    @pytest.mark.grouped_gemm_w8a8
    @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e4m3fnuz])
    def test_grouped_gemm_w8a8_fp8_encodings(self, dtype):
        bits = torch.arange(256, dtype=torch.int16).to(torch.uint8)
        if dtype == torch.float8_e4m3fnuz:
            bits[bits == 128] = 0
        else:
            bits[(bits & 127) == 127] = 0
        args = list(make_grouped_gemm_case([0, 16, 0], 16, 16, flag_gems.device))
        args[0] = bits.view(dtype).reshape(16, 16).to(flag_gems.device)
        args[2] = torch.eye(16).repeat(3, 1, 1).to(dtype).to(flag_gems.device)
        args[1].fill_(1)
        args[3].fill_(1)
        expected = args[0].cpu().float().bfloat16()
        flag_gems.grouped_gemm_w8a8(*args)
        gems_assert_equal(args[4].cpu(), expected)

    @pytest.mark.grouped_gemm_w8a8
    def test_grouped_gemm_w8a8_int8_cancellation(self):
        args = make_grouped_gemm_case(
            [0, 17, 65, 0, 131, 1], 64, 6144, flag_gems.device
        )
        args[0].fill_(127)
        args[2].fill_(127)
        args[0][:, 3072:] = -127
        args[0][:, 2048] = 126
        expected = grouped_gemm_cpu_reference(*args[:4], [0, 17, 65, 0, 131, 1])
        flag_gems.grouped_gemm_w8a8(*args)
        gems_assert_equal(args[4].cpu(), expected)

    @pytest.mark.grouped_gemm_w8a8
    @pytest.mark.parametrize("n,k", [(65, 2051), (1025, 129), (1025, 513)])
    def test_grouped_gemm_w8a8_sparse_changing_groups(self, n, k):
        # Compact/noncompact dispatch boundaries, output/K tails, and a group
        # spanning several M tiles. Reuse metadata storage across different routes.
        lengths = [0] * 257
        lengths[0], lengths[17], lengths[256] = 1, 3, 33
        args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
        for counts in ((1, 3, 33), (33, 1, 3)):
            lengths[0], lengths[17], lengths[256] = counts
            args[6].copy_(
                torch.tensor(lengths, dtype=torch.int32, device=flag_gems.device)
            )
            args[5].copy_(
                torch.repeat_interleave(
                    torch.arange(257, dtype=torch.int32, device=flag_gems.device),
                    args[6],
                    output_size=sum(lengths),
                )
            )
            expected = grouped_gemm_cpu_reference(*args[:4], lengths)
            args[4].fill_(float("nan"))
            assert flag_gems.grouped_gemm_w8a8(*args) is None
            gems_assert_equal(args[4].cpu(), expected)
