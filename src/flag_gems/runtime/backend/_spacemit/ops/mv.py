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

import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems.ops.mv import mv as general_mv
from flag_gems.runtime import torch_device_fn

from ..heuristics_config_utils import (
    HAS_SMT_RVISA,
    _sr_call,
    f16,
    f32,
    smt_rvisa_enabled,
)
from ..heuristics_config_utils import tle as srv

logger = logging.getLogger(__name__)

# The vmt4d arm needs a triton built with smt.vmt4d (spine-triton fork,
# fix/smt-rvisa-const-shadow); the a5 production wheel does not have the
# intrinsic yet, so the arm self-disables there.
HAS_VMT4D = hasattr(smt, "vmt4d")

_VMT4D_BK = 128
_VMT4D_BN = 32


# ---------------------------------------------------------------------------
# vmt4d engine-path GEMV (M=1 matmul). smt.vmt4d lowers to an M=1
# linalg.matmul which spine-mlir's DirectMVPattern (PackMatmul pass,
# main >= cd8e72ff) rewrites to spestruct.vmt4d -> llvm.riscv.smt.vfwmadot,
# using the matrix engine for f16 inputs. Ported from spine-triton
# python/examples/test_smt_mv_vmt4d.py.
#   - f32 acc (engine widening): ~1e-6 rel error; needs spine-mlir with the
#     VMT4D init-pad dtype fix (MR 68, 6f472365);
#   - wt must be a contiguous [K, N] pre-transposed matrix: strided [K, N]
#     views of a [N, K] tensor miscompile (handoff bug 2);
#   - BK=128 x BN=32 is the bench-best tile (mv_bench bench7, 2026-10-27:
#     the [128, 32] f16 tile is 8KB and stays in registers, no stack
#     round-trip for the pack products; [128, 256] = 64KB always spills).
#     The dispatch retains the small-N, large-K shape range from this
#     kernel benchmark when the input already has the required layout.
#   - K % BK == 0 and N % BN == 0 (no tail masking), grid = N / BN <= 512
#     (see the dispatch guard).
# ---------------------------------------------------------------------------
@triton.jit
def _mv_vmt4d_kernel(
    x_ptr,
    wt_ptr,
    y_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    n0 = pid * BN
    r = tl.arange(0, 1)
    ks = tl.arange(0, BK)
    ns = n0 + tl.arange(0, BN)
    acc = tl.zeros((1, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(x_ptr + r[:, None] * K + (k0 + ks)[None, :])  # [1, BK]
        b = tl.load(wt_ptr + (k0 + ks)[:, None] * N + ns[None, :])  # [BK, BN]
        acc = smt.vmt4d(a, b, acc)  # [1, BN]
    tl.store(y_ptr + r[:, None] * N + ns[None, :], acc)


if HAS_SMT_RVISA:

    @srv.raw_kernel
    def _mv_rvisa(
        B: srv.mem(f16),
        A: srv.mem(f16),
        C: srv.mem(f32, out=True),
        K: srv.index,
        row_base: srv.index,
        row_end: srv.index,
    ):
        # streaming svector GEMV: 4 rows per iteration, f32 vmacc
        # accumulation, one horizontal vreduce_sum per row; 1.05-1.08x vs the
        # mm M=1 transposed-b decode path and up to 18x vs the general mv at
        # large N (K3 raw_vs_ops bench, 2026-10-23). Ported from spine-triton
        # python/tests/raw/test_raw_mv_svector.py style2.
        nvl = srv.vconfig(-1, 1)
        # strip-mine: the main loop takes full tiles (fast vload path), the
        # tail loop narrows via vconfig(K-ki) so padded lanes fill with 0 and
        # never affect the accumulation
        Kfloor = (K // nvl) * nvl
        for ni in srv.range(row_base, row_end, 4):
            acc0 = srv.vzero(f32)
            acc1 = srv.vzero(f32)
            acc2 = srv.vzero(f32)
            acc3 = srv.vzero(f32)
            for ki in srv.range(0, Kfloor, nvl):
                va = srv.vload(A, ki)
                vb0 = srv.vload(B, ni * K + ki)
                vb1 = srv.vload(B, (ni + 1) * K + ki)
                vb2 = srv.vload(B, (ni + 2) * K + ki)
                vb3 = srv.vload(B, (ni + 3) * K + ki)
                acc0 = srv.vmacc(acc0, vb0, va)
                acc1 = srv.vmacc(acc1, vb1, va)
                acc2 = srv.vmacc(acc2, vb2, va)
                acc3 = srv.vmacc(acc3, vb3, va)
            # tail loop: runs 0/1 times; independent temporaries so the
            # iter_arg detection does not pick up the main loop's names
            for ki in srv.range(Kfloor, K, nvl):
                nvl = srv.vconfig(K - ki, 1)
                ta = srv.vload(A, ki)
                tb0 = srv.vload(B, ni * K + ki)
                tb1 = srv.vload(B, (ni + 1) * K + ki)
                tb2 = srv.vload(B, (ni + 2) * K + ki)
                tb3 = srv.vload(B, (ni + 3) * K + ki)
                acc0 = srv.vmacc(acc0, tb0, ta)
                acc1 = srv.vmacc(acc1, tb1, ta)
                acc2 = srv.vmacc(acc2, tb2, ta)
                acc3 = srv.vmacc(acc3, tb3, ta)
            srv.sstore(C, ni, srv.vreduce_sum(acc0))
            srv.sstore(C, ni + 1, srv.vreduce_sum(acc1))
            srv.sstore(C, ni + 2, srv.vreduce_sum(acc2))
            srv.sstore(C, ni + 3, srv.vreduce_sum(acc3))

    @triton.jit(do_not_specialize=["K", "N"])
    def _mv_rvisa_host(B, A, C, K, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row_base = pid * BLOCK
        row_end = min(row_base + BLOCK, N)
        _sr_call(_mv_rvisa, outputs=[], inputs=[B, A, C, K, row_base, row_end])


def mv(inp, vec):
    logger.debug("GEMS_SPACEMIT MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, K = inp.shape

    # vmt4d engine path: f16 operands on tile-aligned shapes with a
    # vmt4d-capable triton. Use a [K, N] transpose view only when it is
    # already contiguous, as required by the kernel's pointer arithmetic.
    # Small-N large-K only; N > 512 goes to the svector arm.
    # The grid guard keeps programs within 512 like the svector arm
    # (the runtime silently drops programs beyond that on stacks without
    # the grid-chunked dispatch fix).
    if (
        HAS_VMT4D
        and inp.dtype == torch.float16
        and vec.dtype == torch.float16
        and K > 0
        and N > 0
        and N <= 512
        and K >= 16384
        and K % _VMT4D_BK == 0
        and N % _VMT4D_BN == 0
        and N // _VMT4D_BN <= 512
        and inp.t().is_contiguous()
        and vec.is_contiguous()
    ):
        wt = inp.t()
        y = torch.empty((1, N), dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            _mv_vmt4d_kernel[(N // _VMT4D_BN,)](
                vec.view(1, K), wt, y, K=K, N=N, BK=_VMT4D_BK, BN=_VMT4D_BN
            )
        return y.view(N).to(inp.dtype)

    # The svector kernel reads contiguous f16 operands. N % 4 == 0 keeps
    # its 4-row unroll within the input, and the grid stays within 512
    # programs. Other layouts use the general kernel's stride arguments.
    if (
        HAS_SMT_RVISA
        and smt_rvisa_enabled()
        and inp.dtype == torch.float16
        and vec.dtype == torch.float16
        and N % 4 == 0
        and N > 0
        and inp.is_contiguous()
        and vec.is_contiguous()
    ):
        # BLOCK widens with N so ceil(N / BLOCK) stays within 512 programs;
        # 4 below N<=2048 and 16 above match the benchmarked configs
        block = 4 if N <= 2048 else 16
        while triton.cdiv(N, block) > 512:
            block *= 2
        # the unroll writes whole 4-row groups; pad C to a BLOCK multiple so
        # the final program's stores stay in bounds, then slice [:N]
        Np = ((N + block - 1) // block) * block
        C = torch.empty(Np, dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            _mv_rvisa_host[(Np // block,)](inp, vec, C, K, N, BLOCK=block)
        return C[:N].to(inp.dtype)
    return general_mv(inp, vec)
