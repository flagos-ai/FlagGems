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

"""gcu400 linalg_matmul: the common implementation with the backend GEMMs.

The common ``flag_gems.ops.linalg_matmul`` imports ``flag_gems.ops.mm`` and
``flag_gems.ops.bmm`` directly.  Those generic kernels do not accept the
gcu400 tuned-config keys (``SPLIT_K`` / ``BLOCK_*``), so every call dies with
``KeyError: Keyword argument SPLIT_K was specified but unrecognised``.  The
backend already ships matching ``mm`` / ``bmm`` kernels; this module wires them
into the same shape-folding and autograd logic, and adds the zero-sized fast
exit that a 2-D matmul with an empty M or N needs (a zero grid is not a legal
launch).

The batched arm composes the backend ``bmm`` op, whose hybrid dispatch routes
saturated problems (``B*M*N*K >= 2^29``) to the native primo++ kernel captured
at import time (bit-exact, near-zero dispatch overhead) and keeps the Triton
``bmm_kernel`` below the line; launching the Triton kernel for every size left
the large core cases at 0.51-0.84x of torch (see ``bmm.py`` for the sweep).
"""

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.bmm import bmm
from flag_gems.runtime.backend._enflame.gcu400.ops.mm import mm
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _tiny_mm_kernel(
    A,
    B,
    C,
    N,
    K,
    sa0,
    sa1,
    sb0,
    sb1,
    sc0,
    sc1,
    BLOCK_K: tl.constexpr,
):
    """One program per output element, plain multiply-accumulate.

    The fp32 ``tl.dot`` path fails to link on this backend for 1-row/1-column
    products (``ld.lld: relocation R_DTU_ADDR16_LO_ICALL ... recompile with
    -fPIC`` from the int8 conversion helpers), so those shapes avoid tl.dot.
    """
    pid = tl.program_id(0)
    m = pid // N
    n = pid % N
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        idx = k0 + kk
        mask = idx < K
        a = tl.load(A + m * sa0 + idx * sa1, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + idx * sb0 + n * sb1, mask=mask, other=0.0).to(tl.float32)
        acc += a * b
    tl.store(C + m * sc0 + n * sc1, tl.sum(acc, axis=0).to(C.dtype.element_ty))


def _tiny_mm(a, b):
    M, K = a.shape
    N = b.shape[1]
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    BLOCK_K = max(16, min(triton.next_power_of_2(K), 1024))
    with torch_device_fn.device(a.device):
        _tiny_mm_kernel[(M * N,)](
            a,
            b,
            out,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_K=BLOCK_K,
        )
    return out


def _linalg_bmm(a, b):
    """Batched arm: persistent block_ptr kernel when eligible, else the bmm op.

    bmm runs its Triton kernel at every size, so batched linalg_matmul inherits
    the same deterministic tile tiers a plain bmm call gets. Triton GEMM
    throughput is structurally capped at 145-165 TFLOPS where a saturated
    native pipeline reaches 276-297 (a 10-config sweep found no tuning headroom,
    bmm job j20261111073819_8d5ecfcd; mm/baddbmm/router_gemm alike), and this
    backend stays on Triton there rather than delegating to a native kernel.
    """
    if not (a.is_contiguous() and b.is_contiguous()):
        return bmm(a, b)
    out = _linalg_bmm_persistent(a, b)
    if out is None:
        return bmm(a, b)
    return out


# Hardware DSM budget of one GCU400 cluster (same limit the bmm tile table
# honors): the staged A/B tiles of every software-pipeline stage live there.
_DSM_LIMIT_BYTES = 512 * 1024
# 4 DIEs x 6 SIPs: the same persistent-grid cap the grouped_mm kernel uses.
_PERSISTENT_GRID = 48
# Largest legal launch dimension on this backend (CU_DEVICE_MAX_GRID_SIZE).
_GRID_LIMIT = 65535


@libentry()
@triton.jit
def _linalg_bmm_persistent_kernel(
    A,
    B,
    C,
    Batch,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_bb,
    stride_bk,
    stride_cb,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Batched GEMM with a fixed persistent grid and DMA (block_ptr) tiles.

    One program walks a strided slice of the (batch, M tile, N tile) space, so
    the launch grid stays at the measured GCU400 sweet spot instead of growing
    with the tile count. Loads/stores go through block_ptr, which maps onto
    DTE descriptors (no per-element address math), unlike the raw-pointer
    kernel it replaces.
    """
    num_programs = tl.num_programs(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = grid_m * grid_n
    total_tiles = Batch * tiles_per_batch

    for tile in tl.range(tl.program_id(axis=0), total_tiles, num_programs):
        pid_b = tile // tiles_per_batch
        tile_in_batch = tile % tiles_per_batch
        pid_m = tile_in_batch // grid_n
        pid_n = tile_in_batch % grid_n

        a_block_ptr = tl.make_block_ptr(
            base=A + pid_b * stride_ab,
            shape=(M, K),
            strides=(stride_am, 1),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=B + pid_b * stride_bb,
            shape=(K, N),
            strides=(stride_bk, 1),
            offsets=(0, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            if EVEN:
                a = tl.load(a_block_ptr)
                b = tl.load(b_block_ptr)
            else:
                a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
                b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
            accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

        c_block_ptr = tl.make_block_ptr(
            base=C + pid_b * stride_cb,
            shape=(M, N),
            strides=(stride_cm, 1),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        if EVEN:
            tl.store(c_block_ptr, accumulator.to(C.dtype.element_ty))
        else:
            tl.store(
                c_block_ptr, accumulator.to(C.dtype.element_ty), boundary_check=(0, 1)
            )


def _fit_stages(block_m, block_n, block_k, elem_bytes, num_stages):
    """Shrink the pipeline depth until the staged tiles fit the DSM budget."""
    while num_stages > 1:
        staged = (block_m * block_k + block_k * block_n) * elem_bytes * num_stages
        if staged <= _DSM_LIMIT_BYTES:
            break
        num_stages -= 1
    return num_stages


def _bmm_grid_risk(batch, M, N):
    """True when bmm may run out of launch dimensions on this problem.

    bmm retries with its conservative 64x64 fallback tile whenever the primary
    tile misses the DSM budget at compile time.  That fallback multiplies the
    tile count by up to 64x, and on the model shapes (batch 4, M = 8192,
    N = 128256) the retried grid reached 1026048 against a 65535 limit, so the
    whole matmul died inside the launch.  A persistent launch keeps its grid at
    _PERSISTENT_GRID, so those problems are routed there instead.
    """
    return batch * triton.cdiv(M, 64) * triton.cdiv(N, 64) > _GRID_LIMIT


def _block_ptr_ok(tensor, inner_dim):
    """GCU block_ptr needs a unit inner stride and 16-byte aligned outer strides."""
    align = max(1, 16 // tensor.element_size())
    if tensor.stride(inner_dim) != 1:
        return False
    for dim in range(tensor.dim()):
        if dim != inner_dim and tensor.stride(dim) % align != 0:
            return False
    return True


@lru_cache(maxsize=128)
def _select_persistent_tiles(batch, M, N, K, elem_bytes):
    """Pick the tile for the persistent kernel.

    Saturated problems get the 256x128 tile with the deepest K block that keeps
    two pipeline stages in the DSM budget: on (2,4096,4096,4096) and
    (16,4096,4096,4096) fp16 the 512-deep block measured 0.62x/0.65x of torch
    against 0.27x/0.29x for bmm's 64x64 retry, and it also wins on the 1024 and
    2048 tiers (0.58x/0.58x against 0.57x/0.54x).

    Below that the 128x256 tile is the measured sweet spot whenever the tile
    space can fill the persistent grid; small batches do not, and then the
    128x128 tile doubles the tile count instead of leaving programs idle.

    The K depth is the deepest block that still leaves two pipeline stages in
    the DSM budget: a 256-deep block in fp32 spends 393KB per stage, so fp32 has
    to start at 128 to keep its double buffer (the same limit bmm's table
    honors).
    """
    if M >= 1024 and N >= 1024 and M % 256 == 0 and N % 128 == 0:
        # Deepest K block that still keeps two pipeline stages in the DSM
        # budget. A 512-deep block only fits a single stage in fp16 and
        # measured 2x slower than the 256-deep two-stage tile on the batch-16
        # 1024/2048 shapes (515/3954 us against 260/1955 us), so the depth is
        # taken from the deepest two-stage candidate. Beyond 2048 the 128-deep
        # block wins again (21293 us against 19551 us on batch-16 4096).
        #
        # A batch of one or two is the exception: there the 512-deep single
        # stage still wins on the 4096 tier (1.84 ms against 2.52 ms for the
        # 128-deep tile, both from the persistent kernel), because the deeper
        # block amortizes the K loop instead of adding pipeline pressure.
        if batch <= 2:
            deep_depths = (512, 256, 128) if elem_bytes <= 2 else (256, 128, 64)
            min_stages = 1
        else:
            deep_depths = (256, 128) if max(M, N) <= 2048 else (128, 256)
            if elem_bytes > 2:
                deep_depths = (128, 64)
            min_stages = 2
        for deep_k in deep_depths:
            if K % deep_k != 0:
                continue
            num_stages = _fit_stages(256, 128, deep_k, elem_bytes, 3)
            if num_stages >= min_stages:
                return 256, 128, deep_k, True, num_stages

    wide_tiles = batch * triton.cdiv(M, 128) * triton.cdiv(N, 256)
    block_m, block_n = (128, 256) if wide_tiles >= _PERSISTENT_GRID else (128, 128)
    # Never ask for a tile wider than the problem (oversized tiles waste DSM).
    block_m = max(64, min(block_m, triton.next_power_of_2(M)))
    block_n = max(64, min(block_n, triton.next_power_of_2(N)))

    depths = [256, 128, 64, 32] if elem_bytes <= 2 else [128, 64, 32]
    block_k, num_stages = None, 1
    for candidate_k in depths:
        if candidate_k > max(16, triton.next_power_of_2(K)) or K % candidate_k != 0:
            continue
        stages = _fit_stages(block_m, block_n, candidate_k, elem_bytes, 3)
        block_k, num_stages = candidate_k, stages
        if stages >= 2:
            break
    if block_k is None:
        block_k, num_stages = 32, _fit_stages(block_m, block_n, 32, elem_bytes, 3)
    even = M % block_m == 0 and N % block_n == 0 and K % block_k == 0
    return block_m, block_n, block_k, even, num_stages


def _linalg_bmm_persistent(a, b):
    """Launch the persistent batched kernel; None when the shape is ineligible."""
    batch, M, K = a.shape
    N = b.shape[2]
    # Small M/N tiles on GCU400 are a known-wrong region for the unmasked path
    # (bmm's tile table guards the same corners), and a tiny batch has no tile
    # space to fill; leave those to bmm.
    if M < 64 or N < 64 or K < 16 or batch == 0:
        return None
    # bmm risks a 64x64 retry grid beyond the launch limit on the model shapes;
    # correctness first, so those problems come here even in fp32.
    grid_risk = _bmm_grid_risk(batch, M, N)
    # fp32 otherwise keeps bmm's tier table: the 4-byte tiles fit only two
    # 128-deep stages in DSM, and the block_ptr path measured 1.8x slower than
    # bmm there ((2,4096,4096,4096) fp32: 8.25 ms vs 4.68 ms, 2026-11-21).
    if a.element_size() > 2 and not grid_risk:
        return None
    if not (_block_ptr_ok(a, 2) and _block_ptr_ok(b, 2)):
        return None
    out = torch.empty((batch, M, N), dtype=a.dtype, device=a.device)
    if not _block_ptr_ok(out, 2):
        return None

    block_m, block_n, block_k, even, num_stages = _select_persistent_tiles(
        batch, M, N, K, a.element_size()
    )
    total_tiles = batch * triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    grid = (min(_PERSISTENT_GRID, total_tiles),)
    # No torch_device_fn.device() scope here: the bmm/mm kernels launch directly
    # and that guard costs ~20-30us per call, which dominates the small-batch
    # shapes (2,384,384 runs in ~13us on the device).
    _linalg_bmm_persistent_kernel[grid](
        a,
        b,
        out,
        batch,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        EVEN=even,
        num_warps=4,
        num_stages=num_stages,
    )
    return out


class LinalgMatmulFunction(torch.autograd.Function):
    """Autograd for linalg_matmul.

    Forward -> shape manipulation plus the Triton mm/bmm kernels.
    Backward -> recompute the product in native PyTorch under
    torch.enable_grad, then take gradients with torch.autograd.grad,
    so 1-D folding and batch-dim broadcasting follow native matmul
    gradient semantics.
    """

    @staticmethod
    def forward(ctx, input, other):
        # matmul is only defined for floating point dtypes; reject integral
        # and complex inputs with errors matching the ATen CUDA backend
        # instead of letting them fail inside the Triton kernel.
        if not input.is_floating_point():
            raise NotImplementedError(
                f"linalg_matmul not implemented for '{input.dtype}' on this device"
            )
        if not other.is_floating_point():
            raise NotImplementedError(
                f"linalg_matmul not implemented for '{other.dtype}' on this device"
            )
        if input.dtype != other.dtype:
            raise RuntimeError(
                f"expected mat1 and mat2 to have the same dtype, but got: "
                f"{input.dtype} != {other.dtype}"
            )

        ctx.save_for_backward(input, other)
        return _linalg_matmul_forward(input, other)

    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        orig_dtype = input.dtype
        # run the backward in fp32 for half-precision inputs to preserve
        # gradient precision
        compute_dtype = (
            torch.float32
            if orig_dtype in (torch.float16, torch.bfloat16)
            else orig_dtype
        )
        # the enable_grad block builds a graph for the recomputed forward, so
        # higher-order gradients flow through this backward as well
        with torch.enable_grad():
            input = input.detach().to(compute_dtype).requires_grad_(True)
            other = other.detach().to(compute_dtype).requires_grad_(True)
            output = torch.matmul(input, other)
            grad_input, grad_other = torch.autograd.grad(
                output,
                (input, other),
                grad_output.to(compute_dtype),
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        return (
            None if grad_input is None else grad_input.to(orig_dtype),
            None if grad_other is None else grad_other.to(orig_dtype),
        )


def _empty_matmul_out(shape, dtype, device, fold_first, fold_last):
    out = torch.empty(shape, dtype=dtype, device=device)
    if fold_first:
        out = out.squeeze(-2)
    if fold_last:
        out = out.squeeze(-1)
    return out


def _linalg_matmul_forward(input, other):
    logger.debug("GEMS_ENFLAME LINALG_MATMUL")

    # Fold 1D inputs into 2D per torch.matmul semantics:
    # (K,) @ (K, N) -> (N,) and (M, K) @ (K,) -> (M,), (K,) @ (K,) -> scalar
    fold_first = input.dim() == 1
    fold_last = other.dim() == 1
    if fold_first:
        input = input.unsqueeze(0)
    if fold_last:
        other = other.unsqueeze(1)

    # Handle 2D case: (M, K) @ (K, N) -> (M, N)
    if input.dim() == 2 and other.dim() == 2:
        M, K = input.shape
        N = other.shape[1]
        if M == 0 or N == 0 or K == 0:
            # No work to do, and a Triton launch with a zero grid is invalid.
            return _empty_matmul_out(
                (M, N), input.dtype, input.device, fold_first, fold_last
            )
        if M == 1 or N == 1:
            out = _tiny_mm(input, other)
        else:
            out = mm(input, other)
        if fold_first and fold_last:
            return out.reshape(())
        if fold_first:
            return out[0]
        if fold_last:
            return out[:, 0]
        return out

    # Handle batched cases: broadcast the batch dims, then run a single bmm
    # over the flattened batch. expand produces stride-0 views which the
    # bmm kernel handles natively.
    assert input.dim() >= 2 and other.dim() >= 2, "incompatible dimensions"
    # Same-shape contiguous 3-D inputs need no broadcast/reshape at all; the
    # (expand + broadcast_shapes + 2 reshapes) chain costs more than the kernel
    # itself on small batches, so short-circuit it before touching those APIs.
    if (
        input.dim() == 3
        and other.dim() == 3
        and input.shape[0] == other.shape[0]
        and input.is_contiguous()
        and other.is_contiguous()
        and input.shape[1] > 0
        and input.shape[2] > 0
        and other.shape[2] > 0
    ):
        # _linalg_bmm already returns the (batch, M, N) tensor; no reshape needed.
        return _linalg_bmm(input, other)
    bshape = torch.broadcast_shapes(input.shape[:-2], other.shape[:-2])
    M, K = input.shape[-2], input.shape[-1]
    N = other.shape[-1]
    assert K == other.shape[-2], "incompatible dimensions"
    a = input.expand(bshape + (M, K))
    b = other.expand(bshape + (K, N))
    nbatch = 1
    for s in bshape:
        nbatch *= s
    if nbatch == 0 or M == 0 or N == 0 or K == 0:
        # empty batch / empty matrix: no kernel launch (grid of 0 is invalid),
        # just an empty output with the folded 1-D dims restored
        return _empty_matmul_out(
            bshape + (M, N), input.dtype, input.device, fold_first, fold_last
        )
    out = _linalg_bmm(a.reshape(nbatch, M, K), b.reshape(nbatch, K, N))
    out = out.reshape(bshape + (M, N))
    if fold_first:
        out = out.squeeze(-2)
    if fold_last:
        out = out.squeeze(-1)
    return out


def linalg_matmul(input, other):
    """Matrix product of two tensors.

    The behavior depends on the dimensionality of the inputs:
    - If both inputs are 2D, performs regular matrix multiplication: (M, K) @ (K, N) -> (M, N)
    - If both inputs are 3D (batched), performs batched matrix multiplication:
      (B, M, K) @ (B, K, N) -> (B, M, N)
    - Batch dims are broadcast against each other, and 1D inputs are folded
      into the product per torch.matmul semantics.

    This is an alias for torch.linalg.matmul.
    """
    logger.debug("GEMS_ENFLAME LINALG_MATMUL")
    return LinalgMatmulFunction.apply(input, other)
