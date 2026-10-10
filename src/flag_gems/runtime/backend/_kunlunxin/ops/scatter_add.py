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

"""Kunlunxin (XPU) override for ``scatter_add`` / ``scatter_add_``.

Two tle.raw strategies (no vendor kernel dependency):

1. **cluster_batch_atomic** (SM atomic): 64 cores cooperate per cluster.
   Output tile staged in SM (256KB = 65536 fp32), each core DMA-batches
   idx/src to LM and does ``atomicadd`` on the SM accumulator.  Used when
   ``_use_atomic()`` returns True.

2. **core_batch** (LM per-core): each core owns entire rows, output tile in
   LM (4KB = 1024 fp32), idx/src DMA-batched to LM.  No atomics needed.

3. **Fiber kernel** (fallback for non-contiguous or tle.raw unavailable).

Arbitrary-rank tensors are folded to 2D ``[batch, ylen]`` via dimension
collapsing (vendor ``scatter_calc_common.cpp`` pattern).
"""

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["scatter_add", "scatter_add_"]

# ---------------------------------------------------------------------------
# tle.raw payloads -- fused + dtype-generic, see scatter_add_lm.xpu
# ---------------------------------------------------------------------------
_HAS_TLE_RAW = False
_SUPPORTED_DT = (torch.float32, torch.float16, torch.bfloat16)
_IDX_TAG = {torch.int32: "i32", torch.int64: "i64"}
_DT_TAG = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}
_SA_KERNELS = {}

try:
    import triton.experimental.tle as _tle_ext
    import triton.experimental.tle.language as _tle_lang

    if hasattr(_tle_ext, "raw") and hasattr(_tle_ext.raw, "dialect"):
        _XPU_FILE = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "scatter_add_lm.xpu"
        )

        def _mk_stub(name):
            """Dialect stub whose ``__name__`` is the C++ entry symbol.

            ``RawJITFunction.__init__`` derives the symbol from ``fn.__name__``,
            so the payload names are built from the (path, dtype, idx) table
            instead of 12 hand-written stubs.
            """
            def _payload(y, inp, idx, src, a, b, c, src_rs, idx_rs):
                ...

            _payload.__name__ = name
            return _tle_ext.raw.dialect("xpu3", file=_XPU_FILE)(_payload)

        for _path in ("lm", "atomic"):
            for _dtag in _DT_TAG.values():
                for _itag in _IDX_TAG.values():
                    _nm = f"scatter_add_{_path}_{_dtag}_{_itag}"
                    globals()[_nm] = _mk_stub(_nm)

        # LM path: payload args are (out, inp, idx, src, NROW, NCOL, K, SRC_RS,
        # IDX_RS); the atomic path swaps in (BATCH, IDXLEN, YLEN).
        @triton.jit
        def _sa_lm_f32_i32(Y, INP, IDX, SRC, NROW, NCOL, K, SRC_RS, IDX_RS):
            _tle_lang.raw.call(
                scatter_add_lm_f32_i32,
                (Y, INP, IDX, SRC, NROW, NCOL, K, SRC_RS, IDX_RS),
            )

        @triton.jit
        def _sa_lm_f32_i64(Y, INP, IDX, SRC, NROW, NCOL, K, SRC_RS, IDX_RS):
            _tle_lang.raw.call(
                scatter_add_lm_f32_i64,
                (Y, INP, IDX, SRC, NROW, NCOL, K, SRC_RS, IDX_RS),
            )

        @triton.jit
        def _sa_lm_f16_i32(Y, INP, IDX, SRC, NROW, NCOL, K, SRC_RS, IDX_RS):
            _tle_lang.raw.call(
                scatter_add_lm_f16_i32,
                (Y, INP, IDX, SRC, NROW, NCOL, K, SRC_RS, IDX_RS),
            )

        @triton.jit
        def _sa_lm_f16_i64(Y, INP, IDX, SRC, NROW, NCOL, K, SRC_RS, IDX_RS):
            _tle_lang.raw.call(
                scatter_add_lm_f16_i64,
                (Y, INP, IDX, SRC, NROW, NCOL, K, SRC_RS, IDX_RS),
            )

        @triton.jit
        def _sa_lm_bf16_i32(Y, INP, IDX, SRC, NROW, NCOL, K, SRC_RS, IDX_RS):
            _tle_lang.raw.call(
                scatter_add_lm_bf16_i32,
                (Y, INP, IDX, SRC, NROW, NCOL, K, SRC_RS, IDX_RS),
            )

        @triton.jit
        def _sa_lm_bf16_i64(Y, INP, IDX, SRC, NROW, NCOL, K, SRC_RS, IDX_RS):
            _tle_lang.raw.call(
                scatter_add_lm_bf16_i64,
                (Y, INP, IDX, SRC, NROW, NCOL, K, SRC_RS, IDX_RS),
            )

        @triton.jit
        def _sa_atomic_f32_i32(Y, INP, IDX, SRC, BATCH, IDXLEN, YLEN, SRC_RS, IDX_RS):
            _tle_lang.raw.call(
                scatter_add_atomic_f32_i32,
                (Y, INP, IDX, SRC, BATCH, IDXLEN, YLEN, SRC_RS, IDX_RS),
            )

        @triton.jit
        def _sa_atomic_f32_i64(Y, INP, IDX, SRC, BATCH, IDXLEN, YLEN, SRC_RS, IDX_RS):
            _tle_lang.raw.call(
                scatter_add_atomic_f32_i64,
                (Y, INP, IDX, SRC, BATCH, IDXLEN, YLEN, SRC_RS, IDX_RS),
            )

        @triton.jit
        def _sa_atomic_f16_i32(Y, INP, IDX, SRC, BATCH, IDXLEN, YLEN, SRC_RS, IDX_RS):
            _tle_lang.raw.call(
                scatter_add_atomic_f16_i32,
                (Y, INP, IDX, SRC, BATCH, IDXLEN, YLEN, SRC_RS, IDX_RS),
            )

        @triton.jit
        def _sa_atomic_f16_i64(Y, INP, IDX, SRC, BATCH, IDXLEN, YLEN, SRC_RS, IDX_RS):
            _tle_lang.raw.call(
                scatter_add_atomic_f16_i64,
                (Y, INP, IDX, SRC, BATCH, IDXLEN, YLEN, SRC_RS, IDX_RS),
            )

        @triton.jit
        def _sa_atomic_bf16_i32(Y, INP, IDX, SRC, BATCH, IDXLEN, YLEN, SRC_RS, IDX_RS):
            _tle_lang.raw.call(
                scatter_add_atomic_bf16_i32,
                (Y, INP, IDX, SRC, BATCH, IDXLEN, YLEN, SRC_RS, IDX_RS),
            )

        @triton.jit
        def _sa_atomic_bf16_i64(Y, INP, IDX, SRC, BATCH, IDXLEN, YLEN, SRC_RS, IDX_RS):
            _tle_lang.raw.call(
                scatter_add_atomic_bf16_i64,
                (Y, INP, IDX, SRC, BATCH, IDXLEN, YLEN, SRC_RS, IDX_RS),
            )

        for _path in ("lm", "atomic"):
            for _dt, _dtag in _DT_TAG.items():
                for _it, _itag in _IDX_TAG.items():
                    _SA_KERNELS[(_path, _dt, _it)] = globals()[
                        f"_sa_{_path}_{_dtag}_{_itag}"
                    ]

        _HAS_TLE_RAW = True
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------------------
# Strategy selector (mirrors vendor atomic_add_cases)
# ---------------------------------------------------------------------------
def _use_atomic(batch, idxlen, ylen, itemsize=4):
    """Return True to use the SM atomic path, False for the LM core_batch path."""
    y_size = ylen * itemsize
    if ylen > 0 and ylen / idxlen > 200:
        return False
    if batch < 64:
        return idxlen > 128
    elif batch < 768:
        return y_size >= 8192 and idxlen > 512
    else:
        return y_size > 16384 and idxlen > 2048


# Number of lanes per program.  MUST be 64 -- see module docstring.
LANES = 64
MAX_RANK = 8
_MAX_RANK = tl.constexpr(MAX_RANK)
TARGET_PROGRAMS = 32
MIN_SEGMENT = 16
SPLIT_MEMORY_BUDGET = 64 * 1024 * 1024
COPY_BLOCK = 16384
REDUCE_BLOCK = 1024

_META_CACHE = {}

# The tle.raw payloads are cluster-parallel: each cluster owns a disjoint slice
# of the batch.  Launching grid=(1,) pins the whole op to a single cluster (64
# of 512 cores) -- xpu3 exposes the physical cluster count as
# ``multi_processor_count`` (8 here), matching vendor's ``<<<ctx->ncluster(), 64>>>``.
_CLUSTER_COUNT = {}


def _cluster_count(device):
    key = device.index
    n = _CLUSTER_COUNT.get(key)
    if n is None:
        n = max(1, int(torch.cuda.get_device_properties(device).multi_processor_count))
        _CLUSTER_COUNT[key] = n
    return n


@triton.jit
def _copy_cast_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    """Contiguous copy of ``n_elements``, casting through the pointer dtypes."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(
        dst_ptr + offsets, tl.load(src_ptr + offsets, mask=mask, other=0), mask=mask
    )


@triton.jit
def _copy_cast_kernel_aligned(src_ptr, dst_ptr, BLOCK: tl.constexpr):
    """Same, but for ``numel % BLOCK == 0``: no mask.

    The mask costs ~40% here and, with BLOCK=1024, both variants collapse to
    ~100 GB/s instead of the ~1 TB/s this shape reaches at BLOCK=16384.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets))


@triton.jit
def _reduce_split_kernel(
    acc_ptr,
    inp_ptr,
    out_ptr,
    n_elements,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``out = inp + sum_s acc[s]`` over contiguous private accumulators."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    total = tl.load(inp_ptr + offsets, mask=mask, other=0)
    for s in tl.static_range(SPLIT):
        total += tl.load(acc_ptr + s * n_elements + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, total, mask=mask)


@triton.jit(
    do_not_specialize=[
        "R",
        "K",
        "idx_stride_dim",
        "src_stride_dim",
        "out_stride_dim",
        "acc_numel",
    ]
)
def _scatter_add_fiber_kernel(
    src_ptr,
    index_ptr,
    acc_ptr,
    meta_ptr,
    R,
    K,
    idx_stride_dim,
    src_stride_dim,
    out_stride_dim,
    acc_numel,
    DIM: tl.constexpr,
    RANK: tl.constexpr,
    LANES: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """One lane per fiber, serial walk along ``dim``; no atomics."""
    pid_fiber = tl.program_id(0)
    pid_split = tl.program_id(1)

    lane = pid_fiber * LANES + tl.arange(0, LANES)
    mask = lane < R

    rem = lane.to(tl.int64)
    idx_base = tl.zeros((LANES,), dtype=tl.int64)
    src_base = tl.zeros((LANES,), dtype=tl.int64)
    out_base = tl.zeros((LANES,), dtype=tl.int64)
    for rd in tl.static_range(RANK):
        d = RANK - 1 - rd
        if d != DIM:
            size = tl.load(meta_ptr + d)
            coord = rem % size
            rem = rem // size
            idx_base += coord * tl.load(meta_ptr + _MAX_RANK + d)
            src_base += coord * tl.load(meta_ptr + 2 * _MAX_RANK + d)
            out_base += coord * tl.load(meta_ptr + 3 * _MAX_RANK + d)

    out_base += pid_split * acc_numel

    segment = (K + SPLIT - 1) // SPLIT
    k_begin = pid_split * segment
    k_end = tl.minimum(k_begin + segment, K)
    for k in range(k_begin, k_end):
        index_value = tl.load(
            index_ptr + idx_base + k * idx_stride_dim, mask=mask, other=0
        )
        src_value = tl.load(src_ptr + src_base + k * src_stride_dim, mask=mask, other=0)
        acc = acc_ptr + out_base + index_value * out_stride_dim
        tl.store(acc, tl.load(acc, mask=mask, other=0) + src_value, mask=mask)


def _contiguous_copy(src: torch.Tensor, dst: torch.Tensor) -> None:
    n = dst.numel()
    if n == 0:
        return
    if n % COPY_BLOCK == 0:
        _copy_cast_kernel_aligned[(n // COPY_BLOCK,)](src, dst, BLOCK=COPY_BLOCK)
    else:
        _copy_cast_kernel[(triton.cdiv(n, COPY_BLOCK),)](
            src, dst, n, BLOCK=COPY_BLOCK
        )


def _metadata(index, src_strided, out, dim, device):
    """Device-side shape/stride table for the fiber kernel (cached per layout)."""
    key = (
        tuple(index.shape),
        tuple(index.stride()),
        tuple(src_strided.stride()),
        tuple(out.stride()),
        dim,
        device.index,
    )
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached
    values = [1] * MAX_RANK + [0] * (3 * MAX_RANK)
    rank = index.ndim
    for d in range(rank):
        values[d] = index.shape[d]
        values[MAX_RANK + d] = index.stride(d)
        values[2 * MAX_RANK + d] = src_strided.stride(d)
        values[3 * MAX_RANK + d] = out.stride(d)
    table = torch.tensor(values, dtype=torch.int64, device=device)
    if len(_META_CACHE) > 512:
        _META_CACHE.clear()
    _META_CACHE[key] = table
    return table


def _pick_split(fiber_programs: int, K: int, out_numel: int, itemsize: int) -> int:
    split = max(1, TARGET_PROGRAMS // fiber_programs)
    split = min(split, max(1, K // MIN_SEGMENT))
    while split > 1 and split * out_numel * itemsize > SPLIT_MEMORY_BUDGET:
        split //= 2
    return split


def _scatter_add_fiber(inp, dim, index, src):
    """Fiber-kernel fallback for arbitrary rank / non-contiguous tensors."""
    if inp.ndim > MAX_RANK:
        from flag_gems.ops.scatter_add import scatter_add as _generic

        return _generic(inp, dim, index, src)

    inp_c = inp if inp.is_contiguous() else inp.contiguous()
    out = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)

    K = index.size(dim)
    fibers = index.numel() // K if K else 0
    if fibers == 0 or K == 0 or out.numel() == 0:
        _contiguous_copy(inp_c, out)
        return out

    index_c = index if index.is_contiguous() else index.contiguous()
    src_strided = src.as_strided(index_c.shape, src.stride())

    acc_dtype = (
        torch.float32 if inp.dtype in (torch.float16, torch.bfloat16) else inp.dtype
    )

    fiber_programs = triton.cdiv(fibers, LANES)
    split = _pick_split(
        fiber_programs, K, out.numel(), torch.empty((), dtype=acc_dtype).element_size()
    )

    if split > 1:
        acc = torch.zeros(split * out.numel(), dtype=acc_dtype, device=inp.device)
        target, acc_numel = acc, out.numel()
    elif acc_dtype == inp.dtype:
        acc = out
        _contiguous_copy(inp_c, acc)
        target, acc_numel = acc, 0
    else:
        acc = torch.empty(out.numel(), dtype=acc_dtype, device=inp.device)
        _contiguous_copy(inp_c, acc)
        target, acc_numel = acc, 0

    meta = _metadata(index_c, src_strided, out, dim, inp.device)
    grid = (fiber_programs, split)
    _scatter_add_fiber_kernel[grid](
        src_strided,
        index_c,
        target,
        meta,
        fibers,
        K,
        index_c.stride(dim),
        src_strided.stride(dim),
        out.stride(dim),
        acc_numel,
        DIM=dim,
        RANK=inp.ndim,
        LANES=LANES,
        SPLIT=split,
    )

    if split > 1:
        n = out.numel()
        _reduce_split_kernel[(triton.cdiv(n, REDUCE_BLOCK),)](
            acc, inp_c, out, n, SPLIT=split, BLOCK=REDUCE_BLOCK
        )
    elif acc is not out:
        _contiguous_copy(acc, out)
    return out


# ---------------------------------------------------------------------------
# Dimension folding: arbitrary rank -> 2D [batch, ylen]
# ---------------------------------------------------------------------------
def _can_fold_scatter(inp, dim, index, src):
    """Check whether the tensors can be folded into aligned 2D ``[batch, ylen]``.

    ``inp`` and ``index`` must agree on every non-``dim`` axis so they fold to
    the same ``batch``.  ``src`` may be *larger* than ``index`` on any axis
    (torch semantics: the extra rows are ignored) -- that is exactly the
    benchmark layout, so ``_fold_scatter_dims`` slices it down.
    """
    ndim = inp.ndim
    for d in range(ndim):
        if d == dim:
            continue
        if inp.size(d) != index.size(d):
            return False
    return all(src.size(d) >= index.size(d) for d in range(ndim))


def _fold_scatter_dims(inp, dim, index, src):
    """Fold an arbitrary-rank scatter_add into 2D [batch, ylen].

    Precondition: all tensors share non-dim shapes (checked by _can_fold_scatter).
    ``src`` may be larger than ``index``; it is sliced down to index's shape
    because scatter_add never reads the surplus elements.
    Returns (inp_2d, index_2d, src_2d, ylen, transposed, m, n).
    """
    ndim = inp.ndim
    dim = dim % ndim
    if tuple(src.shape) != tuple(index.shape):
        src = src[tuple(slice(0, index.size(d)) for d in range(ndim))]

    m = 1
    for d in range(dim):
        m *= inp.size(d)
    t_inp = inp.size(dim)
    t_idx = index.size(dim)
    n = 1
    for d in range(dim + 1, ndim):
        n *= inp.size(d)

    if n == 1:
        inp_2d = inp.contiguous().view(m, t_inp)
        idx_2d = index.contiguous().view(m, t_idx)
        src_2d = src.contiguous().view(m, t_idx)
    else:
        # Move scatter axis to last position, then collapse.
        inp_2d = inp.contiguous().view(m, t_inp, n).permute(0, 2, 1).contiguous().view(m * n, t_inp)
        idx_2d = index.contiguous().view(m, t_idx, n).permute(0, 2, 1).contiguous().view(m * n, t_idx)
        src_2d = src.contiguous().view(m, t_idx, n).permute(0, 2, 1).contiguous().view(m * n, t_idx)

    return inp_2d, idx_2d, src_2d, t_inp, n > 1, m, n


def _unfold_result(out_2d, original_shape, dim, m, n):
    """Reverse the dimension fold: 2D [m*n, ylen] -> original shape."""
    t_inp = original_shape[dim]
    if n == 1:
        return out_2d.view(original_shape)
    else:
        out_3d = out_2d.view(m, n, t_inp)
        return out_3d.permute(0, 2, 1).contiguous().view(original_shape)


# ---------------------------------------------------------------------------
# Fused tle.raw driver
# ---------------------------------------------------------------------------
def _launch(path, out, inp, idx, src, a, b, c, src_rs, idx_rs):
    kern = _SA_KERNELS[(path, inp.dtype, idx.dtype)]
    kern[(_cluster_count(inp.device),)](out, inp, idx, src, a, b, c, src_rs, idx_rs)


def _run_2d(out, inp, idx, src, src_rs, idx_rs):
    """One payload launch on contiguous 2D ``[nrow, ncol]`` operands.

    ``out``/``inp`` are ``[nrow, ncol]``, ``idx`` is ``[nrow, k]`` and ``src``
    is row-strided by ``src_rs`` (it may be larger than ``idx`` on either axis;
    the surplus is never read).
    """
    nrow, ncol = inp.shape
    k = idx.shape[1]
    if nrow == 0 or ncol == 0 or k == 0:
        out.copy_(inp)
        return
    if _use_atomic(nrow, k, ncol):
        _launch("atomic", out, inp, idx, src, nrow, k, ncol, src_rs, idx_rs)
    else:
        _launch("lm", out, inp, idx, src, nrow, ncol, k, src_rs, idx_rs)


def _scatter_add_tle(inp, dim, index, src, out):
    """Fused tle.raw path writing into ``out``; False when unsupported."""
    if inp.dtype not in _SUPPORTED_DT or src.dtype != inp.dtype:
        return False
    if index.dtype not in _IDX_TAG:
        return False

    # Fast path: the benchmark layout is already a contiguous 2D [batch, ylen]
    # with the scatter on the last axis, so the payload reads `inp`/`src` in
    # place and writes `out` -- no clone, no contiguous() copy of `src`, and a
    # single launch for the whole op.
    if (
        inp.dim() == 2
        and dim == 1
        and inp.is_contiguous()
        and index.is_contiguous()
        and index.stride(1) == 1
        and src.stride(1) == 1
        and index.size(0) == inp.size(0)
        and src.size(0) >= inp.size(0)
    ):
        _run_2d(out, inp, index, src, src.stride(0), index.stride(0))
        return True

    # General path: fold to 2D [batch, ylen], run, unfold.
    if not _can_fold_scatter(inp, dim, index, src):
        return False
    inp_2d, idx_2d, src_2d, _ylen, _transposed, m, n = _fold_scatter_dims(
        inp, dim, index, src
    )
    if inp_2d.dtype != inp.dtype or idx_2d.dtype != index.dtype:
        return False
    tmp = torch.empty(inp_2d.shape, dtype=inp_2d.dtype, device=inp_2d.device)
    _run_2d(tmp, inp_2d, idx_2d, src_2d, src_2d.stride(0), idx_2d.stride(0))
    out.copy_(_unfold_result(tmp, inp.shape, dim, m, n))
    return True


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------
def scatter_add(inp, dim, index, src):
    logger.debug("GEMS_KUNLUNXIN SCATTER_ADD")
    assert (
        inp.ndim == index.ndim and inp.ndim == src.ndim
    ), "scatter_add: self, index and src must have the same number of dimensions"
    dim = dim % inp.ndim
    assert index.size(dim) <= src.size(dim), "Invalid src"
    for d in range(inp.ndim):
        if d != dim:
            assert index.size(d) <= inp.size(d), "Invalid self"
            assert index.size(d) <= src.size(d), "Invalid src"

    if _HAS_TLE_RAW:
        out = torch.empty_like(inp)
        if _scatter_add_tle(inp, dim, index, src, out):
            return out

    # -- fiber kernel fallback (any rank / non-contiguous) ------------------
    return _scatter_add_fiber(inp, dim, index, src)


def scatter_add_(inp, dim, index, src):
    logger.debug("GEMS_KUNLUNXIN SCATTER_ADD_")
    assert (
        inp.ndim == index.ndim and inp.ndim == src.ndim
    ), "scatter_add_: self, index and src must have the same number of dimensions"
    dim = dim % inp.ndim
    assert index.size(dim) <= src.size(dim), "Invalid src"
    for d in range(inp.ndim):
        if d != dim:
            assert index.size(d) <= inp.size(d), "Invalid self"
            assert index.size(d) <= src.size(d), "Invalid src"

    # In place: hand the payload `inp` itself as the output.  Every row is
    # owned by exactly one core, and the tile is staged (LM/SM) before it is
    # written back, so read-then-write on the same buffer is race free.
    if _HAS_TLE_RAW and _scatter_add_tle(inp, dim, index, src, inp):
        return inp

    out = _scatter_add_fiber(inp, dim, index, src)
    inp.copy_(out)
    return inp
