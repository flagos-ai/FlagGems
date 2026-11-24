import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .bmm import _SAFE_FALLBACK_CONFIG, _clamp_tile, _select_gemm_config

logger = logging.getLogger(__name__)


_SAMPLED_ADDMM_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _broadcast_sparse_csr(input, shape):
    if input.shape == shape:
        return torch.sparse_csr_tensor(
            input.crow_indices().clone(),
            input.col_indices().clone(),
            input.values().clone(),
            size=input.shape,
            dtype=input.dtype,
            device=input.device,
        )
    if input.dim() != 2 or shape[-2:] != input.shape:
        raise RuntimeError(
            f"sparse_sampled_addmm: Cannot broadcast CSR tensor of shape "
            f"{tuple(input.shape)} to {tuple(shape)}"
        )
    M, N = input.shape
    batch_shape = shape[:-2]
    B = math.prod(batch_shape)
    nnz = input._nnz()

    crow = (
        input.crow_indices()
        .unsqueeze(0)
        .expand(B, M + 1)
        .reshape(batch_shape + (M + 1,))
        .contiguous()
    )
    col = (
        input.col_indices()
        .unsqueeze(0)
        .expand(B, nnz)
        .reshape(batch_shape + (nnz,))
        .contiguous()
    )
    val = (
        input.values()
        .unsqueeze(0)
        .expand(B, nnz)
        .reshape(batch_shape + (nnz,))
        .contiguous()
    )
    return torch.sparse_csr_tensor(
        crow, col, val, size=shape, dtype=input.dtype, device=input.device
    )


# GCU limits one grid dimension to 65535 programs.  The dense product kernel is
# launched as a flat (batch, tile) grid over that ceiling, so the tile extent is
# escalated until the launch fits instead of letting a single program iterate over
# several tiles: a dynamic outer loop around ``tl.dot`` keeps the reduction off the
# matrix units and costs two orders of magnitude on the large shapes.
_MAX_GRID_1D = 65535
_MAX_TILE_EXTENT = 256

# Largest fp32 accumulator tile this kernel may use (128 x 256 = 32768 elements = 128 KB).
# The GEMM table is shared with ``bmm``, whose 16-bit tiles are wider because that kernel keeps a
# narrow accumulator; here the accumulator is always fp32, and a wider tile spills.
_MAX_ACC_ELEMS = 32768

# Budget for the batch-wide slot map.  A ``B x M x N`` int32 map removes the
# per-batch launch sequence, but the largest benchmark shapes would need about 1 GB
# for it, so beyond this budget the map is rebuilt once per batch instead.  The
# budget is deliberately below the point where a batch-wide launch starts to lose:
# at 4096^3 the whole batch no longer fits in cache, and one launch over all
# batches measured 2.3x slower than the per-batch sequence (100.6ms vs 41.0ms).
_SLOT_MAP_BUDGET_BYTES = 128 << 20

# Budget for the dense scratch buffer of the two-stage epilogue.  Below it, storing
# the tile densely and folding it onto the values in value order is clearly cheaper
# than one dependent random access per nonzero; above it the dense buffer would
# rival the operator's own footprint, so the slot-map epilogue stays in charge.
_DENSE_EPI_BUDGET_BYTES = 384 << 20


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def sparse_sampled_addmm_kernel(
    mat1_ptr,
    mat2_ptr,
    idx_ptr,
    val_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    # The operand strides are compile-time constants (as in the backend ``bmm``
    # kernel): runtime strides keep the address math opaque and cost fp16/bf16
    # ``tl.dot`` its matrix-unit path.
    stride_mat1_m: tl.constexpr,
    stride_mat1_k: tl.constexpr,
    stride_mat2_k: tl.constexpr,
    stride_mat2_n: tl.constexpr,
    stride_val_b,
    dense_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILES_PER_BATCH: tl.constexpr,
    GROUPED: tl.constexpr,
    DENSE_EPI: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN: tl.constexpr,
    IS_FP64: tl.constexpr,
    MANUAL_DOT: tl.constexpr,
):
    """One output tile of ``mat1 @ mat2``, scattered onto the structural nonzeros.

    ``sampled_addmm`` only keeps the product at the structural nonzeros of the
    sparse operand, so ``idx_ptr`` maps every tile position to its value slot (-1
    outside the pattern) and the tile result is written selectively.  The dense
    product is computed with ``tl.dot`` because it runs on the matrix units: a
    per-nonzero gather formulation is 6x slower here (the gathered ``mat2`` column
    is a scattered access), and the structural sparsity of the benchmark shapes
    (10%) is far too low for a sparse-only sweep to pay off.

    The tile walk mirrors the backend ``bmm`` kernel (``GROUP_M`` swizzle for L2
    reuse, and a static ``EVEN`` branch so the K loop keeps its software pipeline
    instead of degrading ``tl.dot`` to scalar math).

    ``GROUPED`` selects how the batch reaches the operands:

    * ``False`` - the caller launches once per batch and passes already-offset
      pointers.  Nothing batch related is left in the kernel.
    * ``True`` - one launch covers the whole batch and the batch index is folded
      into the *row indices* (``mat1`` seen as ``(B*M, K)``, ``mat2`` as
      ``(B*K, N)``), so the loads stay affine in the offset tensors.  Adding the
      batch offset to the *pointer* instead costs 2.3-2.4x on fp16/bf16 at
      4096^3 (measured: the dot leaves the matrix units), which is why the fold
      is done on the indices and not on the base pointers.
    """
    pid = ext.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if GROUPED:
        pid_b = pid // TILES_PER_BATCH
        pid_local = pid % TILES_PER_BATCH
    else:
        pid_b = 0
        pid_local = pid

    # Grouped ordering: consecutive programs stay inside a GROUP_M-tall row band so
    # the shared ``mat2`` block of that band is reused from cache before eviction.
    width = GROUP_M * num_pid_n
    group_id = pid_local // width
    first_pid_m = group_id * GROUP_M
    group_size = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid_local % group_size)
    pid_n = (pid_local % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m_batch = pid_b * M + offs_m
    offs_k_batch = pid_b * K + offs_k

    a_ptrs = (
        mat1_ptr
        + offs_m_batch[:, None] * stride_mat1_m
        + offs_k[None, :] * stride_mat1_k
    )
    b_ptrs = (
        mat2_ptr
        + offs_k_batch[:, None] * stride_mat2_k
        + offs_n[None, :] * stride_mat2_n
    )

    acc_dtype = tl.float64 if IS_FP64 else tl.float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    # EVEN is a compile-time constant: when M/N/K are all multiples of the selected
    # tiles the masks are dropped entirely, which is what keeps tl.dot on the matrix
    # units (a runtime predicate around the load breaks the pipelined dot).
    if EVEN:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b_val = tl.load(b_ptrs)
            if IS_FP64:
                if MANUAL_DOT:
                    acc += tl.sum(a[:, :, None] * b_val[None, :, :], axis=1)
                else:
                    acc += tl.dot(a, b_val, allow_tf32=False)
            else:
                acc += tl.dot(a, b_val, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_mat1_k
            b_ptrs += BLOCK_K * stride_mat2_k
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b_val = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                other=0.0,
            )
            if IS_FP64:
                if MANUAL_DOT:
                    acc += tl.sum(a[:, :, None] * b_val[None, :, :], axis=1)
                else:
                    acc += tl.dot(a, b_val, allow_tf32=False)
            else:
                acc += tl.dot(a, b_val, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_mat1_k
            b_ptrs += BLOCK_K * stride_mat2_k

    if DENSE_EPI:
        # Write the whole tile: the store is fully coalesced and the value-order
        # gather runs afterwards (see ``_csr_dense_gather_kernel``).  Scattering the
        # tile straight onto the value slots instead costs one dependent random
        # access per nonzero and measured 60% of the runtime on the mid shapes.
        dense_ptrs = dense_ptr + offs_m_batch[:, None] * N + offs_n[None, :]
        if EVEN:
            tl.store(dense_ptrs, acc)
        else:
            tl.store(
                dense_ptrs,
                acc,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            )
    else:
        # Epilogue: only the structural nonzeros of this tile are written back.
        tile_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if EVEN:
            idx = tl.load(idx_ptr + offs_m_batch[:, None] * N + offs_n[None, :])
        else:
            idx = tl.load(
                idx_ptr + offs_m_batch[:, None] * N + offs_n[None, :],
                mask=tile_mask,
                other=-1,
            )
        sampled_mask = idx >= 0

        # Value slots are local to the batch, so only the value vector needs the
        # batch offset (a scalar add on the gather offsets, outside the dot path).
        val_offs = pid_b * stride_val_b + idx
        old = tl.load(val_ptr + val_offs, mask=sampled_mask, other=0.0)
        new = alpha * acc + beta * old.to(acc_dtype)
        tl.store(val_ptr + val_offs, new.to(old.dtype), mask=sampled_mask)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _csr_dense_gather_kernel(
    crow_ptr,
    col_ptr,
    val_ptr,
    dense_ptr,
    alpha,
    beta,
    N,
    rows_per_batch,
    num_batches,
    stride_crow_b,
    stride_col_b,
    stride_val_b,
    BLOCK: tl.constexpr,
):
    """Fold the dense product back onto the CSR values, in value order.

    One program owns whole rows, so the value slots it writes are a contiguous run
    per row (``crow`` segments) and the dense reads stay inside one ``N``-wide row,
    which is cache resident.  This is the counterpart of the ``DENSE_EPI`` store:
    scattering the tile itself onto the slots would issue one dependent random
    access per nonzero, while here the writes are coalesced and only the reads are
    strided.
    """
    pid = ext.program_id(0)
    num_programs = ext.num_programs(0)

    total_rows = rows_per_batch * num_batches
    for gr in tl.range(pid, total_rows, num_programs):
        b = gr // rows_per_batch
        r = gr - b * rows_per_batch
        crow_b = crow_ptr + b * stride_crow_b
        row_start = tl.load(crow_b + r).to(tl.int32)
        row_end = tl.load(crow_b + r + 1).to(tl.int32)
        dense_row = dense_ptr + gr.to(tl.int64) * N
        col_b = col_ptr + b * stride_col_b
        # Value slots are local to the batch, so each batch starts at its own block.
        val_b = val_ptr + b * stride_val_b

        for start in tl.range(row_start, row_end, BLOCK):
            e = start + tl.arange(0, BLOCK)
            mask = e < row_end
            c = tl.load(col_b + e, mask=mask, other=0)
            old = tl.load(val_b + e, mask=mask, other=0.0)
            prod = tl.load(dense_row + c, mask=mask, other=0.0)
            # Same arithmetic order as the fused epilogue: scale/add in the
            # accumulator precision, round once on the store.
            res = alpha * prod + beta * old.to(prod.dtype)
            tl.store(val_b + e, res.to(old.dtype), mask=mask)


@libentry()
@triton.jit
def _csr_value_index_kernel(
    crow_ptr,
    col_ptr,
    idx_ptr,
    N,
    rows_per_batch,
    num_batches,
    stride_crow_b,
    stride_col_b,
    BLOCK: tl.constexpr,
):
    """Scatter the value slot of every structural nonzero into the slot map.

    The map is pre-cleared to -1 by the caller: a bulk elementwise clear runs at
    streaming bandwidth, while clearing row-by-row here was 2x slower overall (the
    per-row stores stay tiny).  Rows of all batches are strided over the grid, so
    the launch stays inside the per-dimension program limit for tall operands and
    the batch loop disappears.
    """
    pid = ext.program_id(0)
    num_programs = ext.num_programs(0)

    total_rows = rows_per_batch * num_batches
    for gr in tl.range(pid, total_rows, num_programs):
        b = gr // rows_per_batch
        r = gr - b * rows_per_batch
        crow_b = crow_ptr + b * stride_crow_b
        row_start = tl.load(crow_b + r).to(tl.int32)
        row_end = tl.load(crow_b + r + 1).to(tl.int32)
        out_base = idx_ptr + gr.to(tl.int64) * N

        for estart in tl.range(row_start, row_end, BLOCK):
            e = estart + tl.arange(0, BLOCK)
            mask = e < row_end
            c = tl.load(col_ptr + b * stride_col_b + e, mask=mask, other=0)
            tl.store(out_base + c, e, mask=mask)


def _dot_block(size, max_block):
    return max(16, min(triton.next_power_of_2(size), max_block))


def _pick_dot_blocks(M, N):
    """Pick ``tl.dot`` extents whose tile count fits one GCU grid dimension."""
    extent = _MAX_TILE_EXTENT
    while True:
        blocks_m = _dot_block(M, min(extent, 64))
        blocks_n = _dot_block(N, extent)
        tiles = triton.cdiv(M, blocks_m) * triton.cdiv(N, blocks_n)
        if tiles <= _MAX_GRID_1D:
            return blocks_m, blocks_n, tiles
        extent *= 2
        if extent > _MAX_GRID_1D:
            raise RuntimeError(
                f"sparse_sampled_addmm: {M}x{N} needs more output tiles than one "
                f"GCU grid dimension allows ({_MAX_GRID_1D})"
            )


def _select_kernel_config(B, M, N, K, dtype):
    """Tile shape for the sampled product.

    The dense product is the same GEMM that ``bmm`` runs, so the sampled addmm
    reuses the backend GEMM tile table (swept per shape tier on GCU400) instead of
    duplicating a second, hand-picked configuration.  Previously this op always
    used a 64-wide M tile with a 64-deep K block, which reached only ~1/5 of the
    ``bmm`` throughput on the saturated shapes.

    The table's 16-bit wide tiles are clamped here: ``bmm`` lets a 16-bit dot keep a
    narrow accumulator, while this kernel always accumulates in fp32 (the dense
    scratch keeps fp32 precision), so a 512x256 tile needs 512 KB of accumulator
    registers, the compiler spills it into 640 KB of local memory (the DSL reports
    ``required localTotalSize:655360``) and the shape collapses.  Measured on
    (16,4096,4096)x(16,4096,4096) with 10% density, same work in both dtypes:
    512x256 fp16 19.90 s vs 128x256 fp32 0.0455 s.
    """
    block_m, block_n, block_k, num_warps, num_stages, even = _select_gemm_config(
        B, M, N, K, dtype
    )
    while block_m * block_n > _MAX_ACC_ELEMS:
        if block_m >= block_n:
            block_m //= 2
        else:
            block_n //= 2
    even = even and M % block_m == 0 and N % block_n == 0
    tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    if tiles > _MAX_GRID_1D:
        # Fall back to wide tiles when a single batch would overflow grid.x.
        block_m, block_n, tiles = _pick_dot_blocks(M, N)
        block_k = _dot_block(K, 64)
        num_warps, num_stages = 4, 2
        even = M % block_m == 0 and N % block_n == 0 and K % block_k == 0
    return block_m, block_n, block_k, num_warps, num_stages, even, tiles


def _sparse_sampled_addmm_impl(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    if input.layout != torch.sparse_csr:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected input to have sparse csr layout, "
            f"but got {input.layout}"
        )
    if mat1.layout != torch.strided:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected mat1 to have strided layout, "
            f"but got {mat1.layout}"
        )
    if mat2.layout != torch.strided:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected mat2 to have strided layout, "
            f"but got {mat2.layout}"
        )
    if out is not None and out.layout != torch.sparse_csr:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected out to have sparse csr layout, "
            f"but got {out.layout}"
        )

    if input.dtype not in _SAMPLED_ADDMM_DTYPES:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected input to be floating-point, "
            f"but got {input.dtype}"
        )
    if input.dtype != mat1.dtype or input.dtype != mat2.dtype:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected all inputs to have the same dtype, "
            f"but got input={input.dtype}, mat1={mat1.dtype}, mat2={mat2.dtype}"
        )

    if input.dense_dim() != 0:
        raise RuntimeError("sparse_sampled_addmm: Expected non-hybrid input tensor")
    if out is not None and out.dense_dim() != 0:
        raise RuntimeError("sparse_sampled_addmm: Expected non-hybrid out tensor")

    if mat1.dim() < 2 or mat2.dim() < 2:
        raise RuntimeError(
            "sparse_sampled_addmm: Expected mat1 and mat2 to be at least 2-D matrices"
        )

    batch_dims = mat1.shape[:-2]
    M, K = mat1.shape[-2:]
    N = mat2.shape[-1]

    if mat2.shape[:-2] != batch_dims:
        raise RuntimeError(
            "sparse_sampled_addmm: Expected mat1 and mat2 to have the same batch size"
        )
    if input.dim() > 2 and input.shape[:-2] != batch_dims:
        raise RuntimeError(
            "sparse_sampled_addmm: Expected input and mat1 to have the same batch size"
        )
    if input.shape[-2] != M or input.shape[-1] != N:
        raise RuntimeError(
            "sparse_sampled_addmm: input.shape[-2:] must match (M, N) of mat1 @ mat2"
        )
    if mat2.shape[-2] != K:
        raise RuntimeError(
            "sparse_sampled_addmm: mat1 and mat2 shapes cannot be multiplied"
        )

    out_shape = batch_dims + (M, N)
    B = math.prod(batch_dims) if batch_dims else 1

    nnz_per_batch = input._nnz()
    nnz = nnz_per_batch * B

    if out is None:
        out = _broadcast_sparse_csr(input, out_shape)
    else:
        if out.shape != out_shape:
            raise RuntimeError(
                f"sparse_sampled_addmm: Expected out shape {out_shape}, got {out.shape}"
            )
        if out._nnz() != nnz_per_batch:
            raise RuntimeError(
                f"sparse_sampled_addmm: Expected out nnz per batch {nnz_per_batch}, "
                f"got {out._nnz()}"
            )
        if out is not input:
            out.copy_(_broadcast_sparse_csr(input, out_shape))

    if mat1.numel() == 0 or mat2.numel() == 0 or nnz == 0 or alpha == 0.0 or K == 0:
        out.values().mul_(beta)
        return out

    mat1_f = mat1.contiguous().reshape(B, M, K)
    mat2_f = mat2.contiguous().reshape(B, K, N)
    val_f = out.values().reshape(B, nnz_per_batch)
    crow_2d = out.crow_indices().reshape(B, M + 1).contiguous()
    col_2d = out.col_indices().reshape(B, nnz_per_batch).contiguous()

    # Tile shape comes from the swept GEMM table (the product is a plain bmm); the
    # reduction is covered by one constexpr K block whenever K is small.
    (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        even,
        tiles_per_batch,
    ) = _select_kernel_config(B, M, N, K, input.dtype)
    is_fp64 = input.dtype == torch.float64
    manual_dot = is_fp64 and runtime_device.vendor_name == "metax"
    if manual_dot:
        BLOCK_M = min(BLOCK_M, 16)
        BLOCK_N = min(BLOCK_N, 16)
        BLOCK_K = min(BLOCK_K, 16)
        even = M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0

    # Two epilogues are available, picked by device-memory budget:
    #   * dense: the tile is stored into a dense ``B*M x N`` scratch buffer (fully
    #     coalesced) and ``_csr_dense_gather_kernel`` folds it onto the values in
    #     value order.  No slot map is needed at all.
    #   * slot map: the tile is scattered straight onto the value slots, which needs
    #     a ``B x M x N`` int32 map (about 1 GB at the largest shapes) plus a clear.
    # The dense scratch keeps the accumulator precision (fp32 for the 16-bit inputs)
    # so the two-stage epilogue rounds exactly once, like the fused one: a bf16
    # scratch rounded the product before ``alpha`` and failed the tight tolerances.
    if input.dtype in (torch.float16, torch.bfloat16):
        dense_dtype = torch.float32
    else:
        dense_dtype = input.dtype
    dense_epi = B * M * N * dense_dtype.itemsize <= _DENSE_EPI_BUDGET_BYTES
    # One launch for the whole batch needs a batch-wide addressable scratch, so it is
    # only possible while the per-batch overhead it saves is worth the footprint.
    batch_grouped = B > 1 and B * M * N * 4 <= _SLOT_MAP_BUDGET_BYTES
    num_batches = B if batch_grouped else 1
    dense_buf = None
    idx_map = None
    if dense_epi:
        dense_buf = torch.empty((B * M, N), dtype=dense_dtype, device=input.device)
    elif batch_grouped:
        idx_map = torch.empty((B, M, N), dtype=torch.int32, device=input.device)
    else:
        idx_map = torch.empty((M, N), dtype=torch.int32, device=input.device)

    grid_fill = (min(M * num_batches, _MAX_GRID_1D),)
    grid_dot = (min(tiles_per_batch * num_batches, _MAX_GRID_1D),)
    grid_gather = (min(B * M, _MAX_GRID_1D),)

    logger.debug(
        "GEMS SPARSE_SAMPLED_ADDMM, [shape info]: batch=%s, M=%s, N=%s, K=%s, "
        "nnz=%s, BLOCK_M=%s, BLOCK_N=%s, BLOCK_K=%s, warps=%s, stages=%s, "
        "even=%s, grouped=%s, dense_epi=%s",
        B,
        M,
        N,
        K,
        nnz,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        even,
        batch_grouped,
        dense_epi,
    )

    batch_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    dot_launch = dict(
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        TILES_PER_BATCH=batch_tiles,
        GROUPED=batch_grouped,
        DENSE_EPI=dense_epi,
        GROUP_M=8,
        EVEN=even,
        IS_FP64=is_fp64,
        MANUAL_DOT=manual_dot,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    def _run_dot(batch_offset, grid, launch):
        # Non-grouped launches slice one batch out (so no batch term reaches the
        # kernel); grouped launches pass the flattened operands, which the kernel
        # addresses through the row indices.
        if batch_grouped:
            mat1_arg, mat2_arg = mat1_f, mat2_f
            val_arg = val_f
            dense_arg = dense_buf
            map_arg = idx_map if idx_map is not None else dense_buf
        else:
            mat1_arg, mat2_arg = mat1_f[batch_offset], mat2_f[batch_offset]
            val_arg = val_f[batch_offset]
            # The dense scratch is indexed by row only, so a per-batch launch takes
            # its own ``M``-row slice of it.
            dense_arg = (
                dense_buf[batch_offset * M :] if dense_buf is not None else idx_map
            )
            map_arg = idx_map if idx_map is not None else dense_arg
        return sparse_sampled_addmm_kernel[grid](
            mat1_arg,
            mat2_arg,
            map_arg,
            val_arg,
            alpha,
            beta,
            M,
            N,
            K,
            mat1_f.stride(1),
            mat1_f.stride(2),
            mat2_f.stride(1),
            mat2_f.stride(2),
            nnz_per_batch,
            dense_arg,
            **launch,
        )

    # Conservative retry used when the swept tiles do not fit the DSM budget.
    fb_m = _clamp_tile(M, _SAFE_FALLBACK_CONFIG[0])
    fb_n = _clamp_tile(N, _SAFE_FALLBACK_CONFIG[1])
    fb_k = _clamp_tile(K, _SAFE_FALLBACK_CONFIG[2], floor=16)
    fb_tiles = triton.cdiv(M, fb_m) * triton.cdiv(N, fb_n)
    fallback_launch = dict(
        BLOCK_M=fb_m,
        BLOCK_N=fb_n,
        BLOCK_K=fb_k,
        TILES_PER_BATCH=fb_tiles,
        GROUPED=batch_grouped,
        DENSE_EPI=dense_epi,
        GROUP_M=8,
        EVEN=(M % fb_m == 0 and N % fb_n == 0 and K % fb_k == 0),
        IS_FP64=is_fp64,
        MANUAL_DOT=manual_dot,
        num_warps=_SAFE_FALLBACK_CONFIG[3],
        num_stages=_SAFE_FALLBACK_CONFIG[4],
    )
    fallback_grid = (min(fb_tiles * num_batches, _MAX_GRID_1D),)

    def _build_slot_map(batch_offset):
        if batch_grouped:
            idx_map.fill_(-1)
            _csr_value_index_kernel[grid_fill](
                crow_2d,
                col_2d,
                idx_map,
                N,
                M,
                B,
                crow_2d.stride(0),
                col_2d.stride(0),
                BLOCK=256,
            )
        else:
            idx_map.fill_(-1)
            _csr_value_index_kernel[grid_fill](
                crow_2d[batch_offset],
                col_2d[batch_offset],
                idx_map,
                N,
                M,
                1,
                crow_2d.stride(0),
                col_2d.stride(0),
                BLOCK=256,
            )

    with torch_device_fn.device(input.device):
        if dense_epi:
            if batch_grouped:
                try:
                    _run_dot(0, grid_dot, dot_launch)
                except triton.runtime.errors.OutOfResources:
                    _run_dot(0, fallback_grid, fallback_launch)
            else:
                for b in range(B):
                    try:
                        _run_dot(b, grid_dot, dot_launch)
                    except triton.runtime.errors.OutOfResources:
                        _run_dot(b, fallback_grid, fallback_launch)
            _csr_dense_gather_kernel[grid_gather](
                crow_2d,
                col_2d,
                val_f,
                dense_buf,
                alpha,
                beta,
                N,
                M,
                B,
                crow_2d.stride(0),
                col_2d.stride(0),
                nnz_per_batch,
                BLOCK=256,
            )
        elif batch_grouped:
            _build_slot_map(0)
            try:
                _run_dot(0, grid_dot, dot_launch)
            except triton.runtime.errors.OutOfResources:
                _run_dot(0, fallback_grid, fallback_launch)
        else:
            for b in range(B):
                _build_slot_map(b)
                try:
                    _run_dot(b, grid_dot, dot_launch)
                except triton.runtime.errors.OutOfResources:
                    _run_dot(b, fallback_grid, fallback_launch)

    return out


def sparse_sampled_addmm(input, mat1, mat2, *, beta=1.0, alpha=1.0):
    logger.debug("GEMS SPARSE_SAMPLED_ADDMM")
    return _sparse_sampled_addmm_impl(input, mat1, mat2, beta=beta, alpha=alpha)


def sparse_sampled_addmm_out(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    logger.debug("GEMS SPARSE_SAMPLED_ADDMM_OUT")
    if out is None:
        raise TypeError("sparse_sampled_addmm(): out must be provided for out variant")
    return _sparse_sampled_addmm_impl(
        input, mat1, mat2, beta=beta, alpha=alpha, out=out
    )
