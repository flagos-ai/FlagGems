import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.linalg_matrix_norm import linalg_matrix_norm
from flag_gems.ops.vector_norm import vector_norm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)


def _parse_ord(ord):
    """Normalize the ord value arriving from aten.

    The aten ``linalg_norm`` schema declares ``Scalar? ord``, and the dispatcher
    passes numeric orders through as their string form (e.g. "2", "-1", "inf").
    Parse those back to floats; keep "fro"/"nuc" as strings.
    """
    if isinstance(ord, str) and ord not in ("fro", "nuc"):
        return float(ord)
    return ord


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit(do_not_specialize=["ord"])
def v_norm_kernel(X, Out, M, N, ord, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Fixed copy of vector_norm.v_norm_kernel (see tests/check_v_norm_kernel.py).

    Fixes over the original:
    1. Masked columns are excluded from the accumulation: with BLOCK_N > N the
       masked lanes load 0.0 and pow(|0.0|, -1) = inf poisons the row sum
       (sum = inf -> out = pow(inf, -1) = 0) for negative ords.
    2. acc_dtype is inferred from the raw pointer BEFORE X becomes a block of
       pointers: afterwards X.type.element_ty returns pointer<fp64>, not fp64,
       so the original fp64 check was always False and fp64 inputs silently
       accumulated in fp32.
    3. Both pow exponents are cast to acc_dtype: `1 / ord` is fp32 in triton
       (int/int division), and fractional ords arrive as fp32 runtime
       scalars, while pow(fp64, fp32) has no libdevice dispatch.
    """
    inp_dtype = X.type.element_ty
    if inp_dtype == tl.float64:
        acc_dtype = tl.float64
    else:
        acc_dtype = tl.float32
    ord = ord.to(acc_dtype)
    pid = ext.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=acc_dtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        a = tl.load(X + cols, mask, other=0.0).to(acc_dtype)
        _sum += tl.where(col_mask, pow(tl.abs(a), ord), 0.0)
    sum = tl.sum(_sum, axis=1)
    out = pow(sum, (1 / ord).to(acc_dtype))[:, None]
    tl.store(Out, out, row_mask)


def _v_norm(x, ord, dim, keepdim, dtype):
    """vector_norm's per-row p-norm path with the fixed v_norm_kernel.

    Mirrors the non-full-reduce branch of vector_norm() for ords that
    dispatch to v_norm_kernel there (any ord other than 2, ±inf, 0 with an
    explicit dim).
    """
    if dtype is not None:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        elif not isinstance(dtype, torch.dtype):
            dtype = torch.float32
    else:
        dtype = x.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16, torch.float64]:
        raise NotImplementedError(f"vector_norm not implemented for {dtype}")

    with torch_device_fn.device(x.device):
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N
        out = torch.empty(shape, dtype=dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        v_norm_kernel[grid](x, out, M, N, ord)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def linalg_norm(A, ord=None, dim=None, keepdim=False, *, dtype=None):
    """Mirror ``torch.linalg.norm`` dispatch (torch routes this op to
    ``linalg_matrix_norm`` or ``linalg_vector_norm``):

    - matrix branch when ``ord`` is "fro"/"nuc", ``dim`` is a 2-tuple, or the
      input is 2D with dim=None (numeric ords then use the matrix norm over
      the last two dims).  Reuses ``linalg_matrix_norm``; dim defaults to
      (-2, -1), ord=None means "fro".
    - vector branch otherwise: ``dim`` as int/1-tuple, or ``dim=None`` which
      flattens the input before applying the vector norm.
      Reuses ``vector_norm``; ord defaults to 2.  The per-row p-norm cases
      (explicit dim, ord other than 2/±inf/0) run the fixed v_norm_kernel
      defined above instead of vector_norm's buggy one.
    """
    logger.debug("GEMS LINALG_NORM")
    ord = _parse_ord(ord)
    if dim is not None:
        dim = [dim] if isinstance(dim, int) else list(dim)
        if len(dim) not in (1, 2):
            raise RuntimeError(
                f"linalg.norm: If dim is specified, it must be of length 1 or 2. "
                f"Got {dim}."
            )
    elif ord is not None:
        if A.ndim not in (1, 2):
            raise RuntimeError(
                "linalg.norm: If dim is not specified but ord is, "
                f"the input must be 1D or 2D. Got {A.ndim}D."
            )
    # Matrix branch: ord='fro'/'nuc', an explicit 2-tuple dim, or a 2D input
    # with dim=None (torch applies the matrix norm over the last two dims
    # there, whether ord is numeric or None).
    if (
        isinstance(ord, str)
        or (dim is not None and len(dim) == 2)
        or (dim is None and A.ndim == 2)
    ):
        # ord=None on the matrix branch means the Frobenius norm.
        return linalg_matrix_norm(
            A,
            "fro" if ord is None else ord,
            (-2, -1) if dim is None else dim,
            keepdim,
            dtype=dtype,
        )
    ord = 2 if ord is None else ord
    # vector_norm's per-row p-norm dispatch (v_norm_kernel): an explicit
    # dim that reduces fewer than all dims, with an ord other than 2/±inf/0.
    # Run the fixed kernel; everything else goes to vector_norm unchanged.
    if (
        dim is not None
        and len(dim) == 1
        and len(dim) < A.ndim
        and ord not in (2, float("inf"), float("-inf"), 0)
    ):
        return _v_norm(A, ord, dim, keepdim, dtype)
    return vector_norm(A, ord, dim, keepdim, dtype=dtype)
