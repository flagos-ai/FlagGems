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
import logging

import pytest
import torch
import triton

import flag_gems

from .accuracy_utils import gems_assert_close, gems_assert_equal

pytestmark = pytest.mark.skipif(
    "grouped_gemm_w8a8" not in vars(flag_gems),
    reason="requires a registered grouped_gemm_w8a8 implementation",
)


def make_grouped_gemm_case(lengths, n, k, device, seed=20261019):
    torch.manual_seed(seed)
    m, e = sum(lengths), len(lengths)
    A = torch.randint(-128, 128, (m, k), dtype=torch.int8, device=device)
    B = torch.randint(-128, 128, (e, n, k), dtype=torch.int8, device=device)
    As = 0.005 + 0.01 * torch.rand((m, 1), device=device)
    Bs = 0.005 + 0.01 * torch.rand((e, n, 1), device=device)
    C = torch.empty((m, n), dtype=torch.bfloat16, device=device)
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
    output = torch.empty((A.shape[0], B.shape[1]), dtype=torch.bfloat16)
    start = 0
    for expert, count in enumerate(lengths):
        end = start + count
        dot = A[start:end].double() @ B[expert].double().T
        output[start:end] = (dot.float() * As[start:end] * Bs[expert, :, 0]).bfloat16()
        start = end
    return output


@pytest.mark.skipif(
    flag_gems.vendor_name != "thead", reason="requires the thead backend"
)
class TestPPUGroupedGEMM:
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
    @pytest.mark.grouped_gemm_w8a8
    def test_grouped_gemm_w8a8(self, lengths, n, k):
        args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
        expected = grouped_gemm_cpu_reference(*args[:4], lengths)
        flag_gems.grouped_gemm_w8a8(*args)
        gems_assert_equal(args[4].cpu(), expected)

    @pytest.mark.parametrize("n,k", [(512, 6144), (6144, 256)])
    @pytest.mark.grouped_gemm_w8a8
    def test_grouped_gemm_w8a8_deepgemm(self, n, k):
        pytest.importorskip("deep_gemm")
        lengths = [1] * 32 + [0] * 224 + [13]
        args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
        expected = torch.empty_like(args[4])
        deepgemm_grouped_gemm(*args[:4], expected, *args[5:])
        flag_gems.grouped_gemm_w8a8(*args)
        gems_assert_close(args[4], expected, torch.bfloat16, atol=1e-3)

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
    @pytest.mark.grouped_gemm_w8a8
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
    @pytest.mark.grouped_gemm_w8a8
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
    flag_gems.vendor_name != "metax", reason="requires the metax backend"
)
class TestMetaXGroupedGEMM:
    @pytest.mark.parametrize(
        "lengths,n,k",
        [
            ([0, 1, 3, 0, 7], 65, 129),
            ([1, 32, 0, 65], 128, 256),
            ([0, 257, 3], 64, 128),
            ([1] + [0] * 255 + [13], 64, 6144),
            ([0, 129, 0, 3968], 257, 2051),
            ([0, 129] + [0] * 254 + [4000], 257, 2051),
            ([0, 1, 0], 17, 1),
            ([0, 0, 0], 64, 128),
        ],
    )
    @pytest.mark.grouped_gemm_w8a8
    def test_grouped_gemm_w8a8(self, lengths, n, k):
        args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
        expected = grouped_gemm_cpu_reference(*args[:4], lengths)
        pointer = args[4].data_ptr()
        assert flag_gems.grouped_gemm_w8a8(*args) is None
        assert args[4].data_ptr() == pointer
        gems_assert_equal(args[4].cpu(), expected)

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
    def test_grouped_gemm_w8a8_changing_expert_rows(self):
        for lengths, n, k in (
            ([0, 257, 0, 511], 64, 128),
            ([256, 0, 512, 0], 128, 256),
            ([256, 0, 512, 0], 64, 128),
            ([0, 257, 0, 511], 128, 256),
        ):
            args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
            expected = grouped_gemm_cpu_reference(*args[:4], lengths)
            flag_gems.grouped_gemm_w8a8(*args)
            gems_assert_equal(args[4].cpu(), expected)

    @pytest.mark.grouped_gemm_w8a8
    @pytest.mark.parametrize(
        "dtype", [torch.int8, torch.float8_e4m3fn, torch.float8_e4m3fnuz]
    )
    @pytest.mark.parametrize(
        "lengths,n,k",
        [
            ([0, 3, 1], 17, 0),
            ([0, 3, 1], 0, 16),
            ([0, 0], 16, 16),
            ([0, 17, 65, 0, 131, 1], 65, 129),
            ([0, 257, 3], 128, 256),
            ([0, 513, 0], 256, 256),
        ],
    )
    def test_grouped_gemm_w8a8_dtypes_tails(self, lengths, n, k, dtype):
        args = list(make_grouped_gemm_case(lengths, n, k, flag_gems.device))
        if dtype != torch.int8:
            for index in (0, 2):
                args[index] = (
                    (args[index].cpu().float() / 64).to(dtype).to(flag_gems.device)
                )
        expected = grouped_gemm_cpu_reference(*args[:4], lengths)
        args[4].fill_(float("nan"))
        assert flag_gems.grouped_gemm_w8a8(*args) is None
        if dtype == torch.int8:
            gems_assert_equal(args[4].cpu(), expected)
        else:
            # FP32 dot reduction order may change BF16 rounding near cancellation.
            # Permit one BF16 relative rounding step, with no absolute tolerance.
            torch.testing.assert_close(
                args[4].cpu(), expected, rtol=torch.finfo(torch.bfloat16).eps, atol=0
            )

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


@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="requires the Hygon implementation"
)
class TestHygonGroupedGEMM:
    test_grouped_gemm_w8a8 = TestPPUGroupedGEMM.test_grouped_gemm_w8a8
    test_grouped_gemm_w8a8_int8_extremes = (
        TestPPUGroupedGEMM.test_grouped_gemm_w8a8_int8_extremes
    )
    test_grouped_gemm_w8a8_pipeline = TestPPUGroupedGEMM.test_grouped_gemm_w8a8_pipeline
    test_grouped_gemm_w8a8_int8_cancellation = (
        TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_int8_cancellation
    )
    test_grouped_gemm_w8a8_sparse_changing_groups = (
        TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_sparse_changing_groups
    )

    @pytest.mark.parametrize(
        "lengths,n,k",
        [
            ([1, 2, 0, 3, 0, 0], 512, 6144),
            ([1, 2, 0, 3, 0, 0], 6144, 256),
            ([2, 1], 512, 6144),
        ],
    )
    @pytest.mark.grouped_gemm_w8a8
    def test_grouped_gemm_w8a8_specialized_paths(self, lengths, n, k):
        args = make_grouped_gemm_case(lengths, n, k, flag_gems.device)
        expected = grouped_gemm_cpu_reference(*args[:4], lengths)
        assert flag_gems.grouped_gemm_w8a8(*args) is None
        gems_assert_equal(args[4].cpu(), expected)

    @pytest.mark.parametrize("lengths", [[5, 0, 0, 0, 0], [1, 5, 0, 0, 0, 0]])
    @pytest.mark.grouped_gemm_w8a8
    def test_grouped_gemm_w8a8_gemv_long_runs(self, lengths):
        module = importlib.import_module(flag_gems.grouped_gemm_w8a8.__module__)
        if (
            not module._HAS_GEMV
            or triton.runtime.driver.active.get_current_target().arch != "gfx936"
        ):
            pytest.skip("requires gfx936 with Hygon Gluon GEMV support")
        args = make_grouped_gemm_case(lengths, 512, 6144, flag_gems.device)
        expected = grouped_gemm_cpu_reference(*args[:4], lengths)
        assert flag_gems.grouped_gemm_w8a8(*args) is None
        gems_assert_equal(args[4].cpu(), expected)

    @pytest.mark.grouped_gemm_w8a8
    def test_grouped_gemm_w8a8_split_k_fallback(self, monkeypatch, caplog):
        op = flag_gems.grouped_gemm_w8a8
        module = importlib.import_module(op.__module__)
        monkeypatch.setattr(module, "_HAS_GEMV", False)

        lengths = [1, 2, 0, 3, 0, 0]
        args = make_grouped_gemm_case(lengths, 512, 6144, flag_gems.device)
        expected = grouped_gemm_cpu_reference(*args[:4], lengths)
        with caplog.at_level(logging.DEBUG, logger=module.__name__):
            assert op(*args) is None

        assert (
            module.__name__,
            logging.DEBUG,
            "GEMS_HYGON GROUPED_GEMM_W8A8",
        ) in caplog.record_tuples
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
