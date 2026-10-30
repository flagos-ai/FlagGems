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
import triton.language as tl

import flag_gems

from . import accuracy_utils as utils

pytestmark = [
    pytest.mark.mm,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend", reason="Ascend launch-grid regression"
    ),
]


@pytest.fixture(scope="module")
def ascend_mm():
    return importlib.import_module("flag_gems.runtime.backend._ascend.ops.mm")


def _run_kernel(ascend_mm, a, b, out, split_k):
    # Exercise the smallest tuning candidate directly: autotuning may select
    # a larger tile and otherwise hide the overflowing-grid regression. Keep
    # the heuristics wrapper so all derived constexprs (including divisibility
    # flags) are supplied exactly as they are for a normal launch.
    kernel = ascend_mm.mm_kernel
    while not isinstance(kernel, triton.runtime.Heuristics):
        kernel = kernel.fn
    M, K = a.shape
    N = b.shape[1]
    meta = {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": split_k}
    grid = ascend_mm._mm_grid(M, N, meta)
    assert len(grid) == 1 and 0 < grid[0] <= 65535
    kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        *a.stride(),
        *b.stride(),
        *out.stride(),
        dot_out_dtype=tl.float32,
        GROUP_M=8,
        **meta,
    )


@pytest.mark.parametrize(
    "M, N, K, split_k",
    [
        (4080, 8224, 32, 1),  # Exactly 65535 logical programs.
        (4096, 8192, 33, 1),  # 65536 programs; one tile after the first pass.
        (4352, 10240, 64, 1),  # 87040 programs, matching the reported coreDim.
        (4096, 4096, 65, 2),  # The split-K factor alone crosses the limit.
    ],
)
def test_mm_ascend_core_dim_boundary(ascend_mm, M, N, K, split_k):
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
    b = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
    # Use FP32 for split-K to avoid rounding each partial sum to FP16 before
    # the atomic add; this test checks launch limits and tile coverage.
    out_dtype = torch.float32 if split_k > 1 else a.dtype
    out = torch.zeros((M, N), dtype=out_dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())
    _run_kernel(ascend_mm, a, b, out, split_k)
    utils.gems_assert_close(out, ref, out.dtype, reduce_dim=K)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("split_k, K", [(1, 0), (1, 64), (1, 71), (2, 33), (2, 129)])
def test_mm_ascend_repeated_tiles(ascend_mm, monkeypatch, dtype, split_k, K):
    # Force many iterations with small allocations; also cover a partial
    # GROUP_M, masked M/N/K tails and transposed inputs.
    monkeypatch.setattr(ascend_mm, "_MAX_CORE_DIM", 3)
    torch.manual_seed(0)
    M, N = 145, 97
    a = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    b = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    out_dtype = torch.float32 if split_k > 1 else dtype
    out = torch.zeros((M, N), dtype=out_dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())
    _run_kernel(ascend_mm, a, b, out, split_k)
    utils.gems_assert_close(out, ref, out.dtype, reduce_dim=max(K, 1))


@pytest.mark.parametrize("use_out", [False, True])
def test_mm_ascend_large_grid_entrypoints(ascend_mm, use_out):
    torch.manual_seed(0)
    M, N, K = 4352, 10240, 33
    a = torch.randn((M, K), dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.randn((N, K), dtype=torch.bfloat16, device=flag_gems.device).t()
    ref = torch.mm(utils.to_reference(a).float(), utils.to_reference(b).float())
    if use_out:
        out = torch.empty((M, N), dtype=a.dtype, device=a.device)
        assert ascend_mm.mm_out(a, b, out=out) is out
    else:
        out = ascend_mm.mm(a, b)
    utils.gems_assert_close(out, ref, a.dtype, reduce_dim=K)
