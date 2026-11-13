import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# torch GCU sparse-CSR accessor shim (torch dispatch gap workaround)
#
# torch_gcu registers no SparseCsr kernels for its PrivateUse1 backend, so
# every member accessor on a GCU sparse CSR tensor fails inside torch's
# dispatcher before FlagGems code even runs:
#   - aten::crow_indices / col_indices / values fall through to the
#     CompositeExplicitAutograd error shims (unconditional TORCH_CHECK for
#     backends without SparseCsr kernels, e.g. "crow_indices expected sparse
#     row compressed tensor layout but got SparseCsr");
#   - aten::_nnz and aten::_copy_from have no kernel at all
#     (NotImplementedError, 'SparseCsrgcu' backend).
# The C++ SparseCsrTensorImpl keeps the member tensors, but Python cannot
# reach them, so this module (the FlagGems op that consumes GCU CSR inputs)
# installs a narrow Python-level bridge at import time:
#   1. torch.sparse_csr_tensor is wrapped: when the result lands on the GCU
#      it remembers the exact (crow, col, values) tensors it was built from
#      (members are pre-moved to the target device first, mirroring the
#      factory's "device kwarg converts the inputs" semantics);
#   2. torch.Tensor.crow_indices / col_indices / values / _nnz / dense_dim
#      are wrapped to serve the stashed members for stashed GCU CSR tensors
#      and fall through to whatever layer was installed before (usually the
#      original TensorBase method), so CPU / COO / hybrid paths behave
#      exactly as before and sibling vendor shims (e.g. the sparse_bsr_tensor
#      P21 fix) compose in either import order.
# The impl below additionally copies CSR members member-wise instead of
# aten::copy_ (no SparseCsrgcu kernel). Everything under the metadata layer
# (the Triton kernels) already runs on plain dense GCU tensors.
# ---------------------------------------------------------------------------
_CSR_MEMBER_ATTR = "_flaggems_gcu_csr_members"
_SHIM_MARKER = "_flaggems_gcu_csr_shim"
# GCU hardware cap on the x dimension of a kernel launch grid.
_MAX_GRID_X = 65535


def _stashed_csr_members(tensor):
    if isinstance(tensor, torch.Tensor):
        return getattr(tensor, _CSR_MEMBER_ATTR, None)
    return None


def _forward_csr_factory(
    factory, crow_indices, col_indices, values, size, args, kwargs
):
    """Replay a ``sparse_csr_tensor`` call with the caller's exact call shape.

    The native constructor has a 4-argument overload whose positional
    ``size`` must be a real sequence and a 3-argument overload that infers
    the dense extents; forwarding a positional ``None`` is rejected by both
    (``TypeError: got (Tensor, Tensor, Tensor, NoneType)``), so the size
    slot is dropped whenever it is absent. A ``size`` supplied as a keyword
    keeps travelling untouched inside ``kwargs``.
    """
    if size is None:
        return factory(crow_indices, col_indices, values, *args, **kwargs)
    return factory(crow_indices, col_indices, values, size, *args, **kwargs)


def _install_gcu_sparse_csr_accessor_shim():
    if getattr(torch.Tensor.crow_indices, _SHIM_MARKER, False):
        return

    device_type = runtime_device.name
    # Chain onto whatever currently sits on torch.Tensor (the raw TensorBase
    # method, or a sibling vendor shim such as the sparse_bsr_tensor P21
    # fix): un-stashed tensors keep the previous behavior exactly, so the
    # two shims compose in either import order.
    orig_crow = torch.Tensor.crow_indices
    orig_col = torch.Tensor.col_indices
    orig_values = torch.Tensor.values
    orig_nnz = torch.Tensor._nnz
    orig_dense_dim = torch.Tensor.dense_dim

    def crow_indices(self):
        members = getattr(self, _CSR_MEMBER_ATTR, None)
        if members is not None:
            return members[0]
        return orig_crow(self)

    def col_indices(self):
        members = getattr(self, _CSR_MEMBER_ATTR, None)
        if members is not None:
            return members[1]
        return orig_col(self)

    def values(self):
        members = getattr(self, _CSR_MEMBER_ATTR, None)
        if members is not None:
            return members[2]
        return orig_values(self)

    def _nnz(self):
        members = getattr(self, _CSR_MEMBER_ATTR, None)
        if members is not None:
            # ATen reports per-batch nnz: values shape is batch + (nnz,) +
            # dense, crow shape is batch + (M + 1), so nnz sits at index
            # crow.dim() - 1 of values.
            crow, _, vals = members
            return vals.shape[crow.dim() - 1]
        return orig_nnz(self)

    def dense_dim(self):
        members = getattr(self, _CSR_MEMBER_ATTR, None)
        if members is not None:
            # values.dim() = batch + 1 + dense_dim, crow.dim() = batch + 1.
            crow, _, vals = members
            return vals.dim() - crow.dim()
        return orig_dense_dim(self)

    for method_name, patched in (
        ("crow_indices", crow_indices),
        ("col_indices", col_indices),
        ("values", values),
        ("_nnz", _nnz),
        ("dense_dim", dense_dim),
    ):
        setattr(patched, _SHIM_MARKER, True)
        setattr(torch.Tensor, method_name, patched)

    orig_factory = torch.sparse_csr_tensor

    def _already_on(tensor, target):
        if tensor.device.type != target.type:
            return False
        return target.index is None or tensor.device == target

    def sparse_csr_tensor(
        crow_indices, col_indices, values, size=None, *args, **kwargs
    ):
        device = kwargs.get("device")
        if (
            args
            or device is None
            or not (
                isinstance(crow_indices, torch.Tensor)
                and isinstance(col_indices, torch.Tensor)
                and isinstance(values, torch.Tensor)
            )
        ):
            return _forward_csr_factory(
                orig_factory, crow_indices, col_indices, values, size, args, kwargs
            )
        try:
            target = torch.device(device)
        except (TypeError, ValueError):
            return _forward_csr_factory(
                orig_factory, crow_indices, col_indices, values, size, args, kwargs
            )
        if target.type != device_type:
            return _forward_csr_factory(
                orig_factory, crow_indices, col_indices, values, size, args, kwargs
            )
        # Pre-move the members so the factory (and the stashed view) see the
        # exact same tensors ("to" is a no-op that returns self when the
        # tensor already sits on the target device).
        crow = (
            crow_indices
            if _already_on(crow_indices, target)
            else crow_indices.to(target)
        )
        col = (
            col_indices if _already_on(col_indices, target) else col_indices.to(target)
        )
        val = values if _already_on(values, target) else values.to(target)
        dtype = kwargs.get("dtype")
        if dtype is not None and val.dtype != dtype:
            # The factory would replace the values tensor with a converted
            # copy; leave the call unstashed rather than serve stale members.
            return _forward_csr_factory(
                orig_factory, crow_indices, col_indices, values, size, args, kwargs
            )
        result = _forward_csr_factory(orig_factory, crow, col, val, size, (), kwargs)
        if result.device.type == device_type and result.layout == torch.sparse_csr:
            result._flaggems_gcu_csr_members = (crow, col, val)
        return result

    setattr(sparse_csr_tensor, _SHIM_MARKER, True)
    torch.sparse_csr_tensor = sparse_csr_tensor


_install_gcu_sparse_csr_accessor_shim()


def _copy_csr_members_(out, src):
    """Copy sparse CSR ``src`` into ``out``.

    aten::copy_ has no SparseCsrgcu kernel, so on the GCU path the three
    member tensors are copied member-wise (plain dense copies); every other
    backend keeps the native ``out.copy_(src)`` semantics.
    """
    out_members = _stashed_csr_members(out)
    src_members = _stashed_csr_members(src)
    if out_members is not None and src_members is not None:
        out_members[0].copy_(src_members[0])
        out_members[1].copy_(src_members[1])
        out_members[2].copy_(src_members[2])
        return
    out.copy_(src)


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


@libentry()
@triton.jit
def _csr_value_index_kernel(
    crow_ptr,
    col_ptr,
    idx_ptr,
    M,
    N,
    nnz_per_batch,
    total_rows,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    num_progs = tl.num_programs(0)
    # GCU caps grid.x at 65535, so the (batch, row) pairs are strided over the
    # available programs instead of the generic one-program-per-row grid
    # (B * M reaches 65536 for the core benchmark shapes).
    for pid_flat in range(pid, total_rows, num_progs):
        b = pid_flat // M
        r = pid_flat % M

        crow_base = crow_ptr + b.to(tl.int64) * (M + 1)
        row_start = tl.load(crow_base + r).to(tl.int64)
        row_end = tl.load(crow_base + r + 1).to(tl.int64)

        col_base = col_ptr + b.to(tl.int64) * nnz_per_batch
        out_base = idx_ptr + b.to(tl.int64) * M * N + r.to(tl.int64) * N

        for start in range(row_start, row_end, BLOCK):
            e = start + tl.arange(0, BLOCK)
            mask = e < row_end
            c = tl.load(col_base + e, mask=mask, other=0)
            tl.store(out_base + c, e.to(tl.int32), mask=mask)


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
    nnz_per_batch,
    total_tiles,
    stride_mat1_b,
    stride_mat1_m,
    stride_mat1_k,
    stride_mat2_b,
    stride_mat2_k,
    stride_mat2_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_FP64: tl.constexpr,
    MANUAL_DOT: tl.constexpr,
):
    pid = ext.program_id(0)
    num_progs = tl.num_programs(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = tl.cdiv(M, BLOCK_M) * num_pid_n
    # GCU caps grid.x at 65535, so the (batch, tile) pairs are strided over
    # the available programs instead of the generic one-program-per-tile grid
    # (B * tiles reaches 65536 for the core benchmark shapes).
    for pid_flat in range(pid, total_tiles, num_progs):
        b = pid_flat // num_tiles
        tile_id = pid_flat % num_tiles
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        b64 = b.to(tl.int64)
        a_ptrs = (
            mat1_ptr
            + b64 * stride_mat1_b
            + offs_m[:, None] * stride_mat1_m
            + offs_k[None, :] * stride_mat1_k
        )
        b_ptrs = (
            mat2_ptr
            + b64 * stride_mat2_b
            + offs_k[:, None] * stride_mat2_k
            + offs_n[None, :] * stride_mat2_n
        )

        acc_dtype = tl.float64 if IS_FP64 else tl.float32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K),
                other=0.0,
            )
            b_val = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N),
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

        offs_m64 = offs_m.to(tl.int64)
        offs_n64 = offs_n.to(tl.int64)
        tile_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        idx = tl.load(
            idx_ptr + b64 * M * N + offs_m64[:, None] * N + offs_n64[None, :],
            mask=tile_mask,
            other=-1,
        )
        sampled_mask = idx >= 0

        val_offs = b64 * nnz_per_batch + idx.to(tl.int64)
        old = tl.load(val_ptr + val_offs, mask=sampled_mask, other=0.0)
        new = alpha * acc + beta * old.to(acc_dtype)
        tl.store(val_ptr + val_offs, new.to(old.dtype), mask=sampled_mask)


def _dot_block(size, max_block):
    return max(16, min(triton.next_power_of_2(size), max_block))


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
            _copy_csr_members_(out, _broadcast_sparse_csr(input, out_shape))

    if mat1.numel() == 0 or mat2.numel() == 0 or nnz == 0 or alpha == 0.0 or K == 0:
        out.values().mul_(beta)
        return out

    mat1_f = mat1.contiguous().reshape(B, M, K)
    mat2_f = mat2.contiguous().reshape(B, K, N)
    val_f = out.values().reshape(B * nnz_per_batch)
    crow_2d = out.crow_indices().reshape(B, M + 1).contiguous()
    col_2d = out.col_indices().reshape(B, nnz_per_batch).contiguous()

    idx_map = torch.full((B, M * N), -1, dtype=torch.int32, device=input.device)

    BLOCK_M = _dot_block(M, 64)
    BLOCK_N = _dot_block(N, 64)
    BLOCK_K = _dot_block(K, 32)

    manual_dot = input.dtype == torch.float64 and runtime_device.vendor_name == "metax"
    if manual_dot:
        BLOCK_M = min(BLOCK_M, 16)
        BLOCK_N = min(BLOCK_N, 16)
        BLOCK_K = min(BLOCK_K, 16)

    # GCU caps grid.x at 65535: the kernels stride (batch, row) / (batch,
    # tile) pairs over a capped program count, so B * M and B * tiles of any
    # size stay launchable (the generic one-per-row/tile grid reaches 65536
    # for the core benchmark shapes and trips OutOfResources).
    total_rows = B * M
    grid_fill = (min(total_rows, _MAX_GRID_X),)
    total_tiles = B * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(total_tiles, _MAX_GRID_X),)

    logger.debug(
        "GEMS SPARSE_SAMPLED_ADDMM, [shape info]: batch=%s, M=%s, N=%s, K=%s, "
        "nnz=%s, BLOCK_M=%s, BLOCK_N=%s, BLOCK_K=%s",
        B,
        M,
        N,
        K,
        nnz,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    with torch_device_fn.device(input.device):
        _csr_value_index_kernel[grid_fill](
            crow_2d,
            col_2d,
            idx_map,
            M,
            N,
            nnz_per_batch,
            total_rows,
            BLOCK=256,
        )
        sparse_sampled_addmm_kernel[grid](
            mat1_f,
            mat2_f,
            idx_map,
            val_f,
            alpha,
            beta,
            M,
            N,
            K,
            nnz_per_batch,
            total_tiles,
            mat1_f.stride(0),
            mat1_f.stride(1),
            mat1_f.stride(2),
            mat2_f.stride(0),
            mat2_f.stride(1),
            mat2_f.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            IS_FP64=input.dtype == torch.float64,
            MANUAL_DOT=manual_dot,
            num_warps=4,
        )

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
