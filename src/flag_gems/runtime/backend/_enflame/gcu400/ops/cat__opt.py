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

"""dim=0 linear fast path for cat (pure helper, no module binding).

KernelGen `cat_copy_func_kernel_4` performs 3 int64 divisions / modulos per element.
With dim==0, pre_idx is always 0 and (idx//P)*P + idx%P == idx, so
    out_idx == idx + the starting element offset of that input within the output
can drop coordinate decomposition entirely. Probe measured 2.1x (small) ~ 26.7x (large), bit-identical to torch.cat.

Called from `cat()` in `cat.py` inside the function body; returns None when the strict boundaries are not met,
so the caller falls back to the original full logic (float8 / non-zero dim / >4 inputs / dtype mismatch /
non-contiguous / shape mismatch outside the cat dimension).
"""

import torch
import triton
import triton.language as tl

from ..utils import fast_launch

__all__ = ["try_fast_cat", "try_fast_cat_out"]

_GRID_CAP = 48
_BLOCK_EVEN = 16384
_BLOCK_MASKED = 8192
_MAX_WORDS = 2**31 - 1 - _BLOCK_EVEN
_TILE_TARGET = 16384


@triton.jit
def _linear_cat_kernel(
    Out,
    In0,
    In1,
    In2,
    In3,
    n0,
    n1,
    n2,
    n3,
    o0,
    o1,
    o2,
    o3,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    off = tl.arange(0, BLOCK)
    idx = pid_x * BLOCK + off
    if pid_y == 0:
        p = In0
    elif pid_y == 1:
        p = In1
    elif pid_y == 2:
        p = In2
    else:
        p = In3
    n = tl.where(pid_y == 0, n0, tl.where(pid_y == 1, n1, tl.where(pid_y == 2, n2, n3)))
    o = tl.where(pid_y == 0, o0, tl.where(pid_y == 1, o1, tl.where(pid_y == 2, o2, o3)))
    mask = (idx < n) & (pid_y < NT)
    d = tl.load(p + idx, mask=mask, other=0)
    tl.store(Out + o + idx, d, mask=mask)


@triton.jit
def _flat_cat_kernel(
    Out,
    In0,
    In1,
    In2,
    In3,
    n0,
    n1,
    n2,
    n3,
    o0: tl.int64,
    o1: tl.int64,
    o2: tl.int64,
    o3: tl.int64,
    num_tiles,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    GRID: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    if pid_y == 0:
        src = In0
    elif pid_y == 1:
        src = In1
    elif pid_y == 2:
        src = In2
    else:
        src = In3
    o = tl.where(pid_y == 0, o0, tl.where(pid_y == 1, o1, tl.where(pid_y == 2, o2, o3)))
    n = tl.where(pid_y == 0, n0, tl.where(pid_y == 1, n1, tl.where(pid_y == 2, n2, n3)))
    dst = Out + o
    for t in tl.range(pid_x, num_tiles, GRID, num_stages=2):
        idx = t * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            val = tl.load(src + idx)
            tl.store(dst + idx, val)
        else:
            m = idx < n
            val = tl.load(src + idx, mask=m, other=0)
            tl.store(dst + idx, val, mask=m)


@triton.jit
def _blocked_cat_kernel(
    Out,
    In0,
    In1,
    In2,
    In3,
    R,
    c0,
    c1,
    c2,
    c3,
    o0: tl.int64,
    o1: tl.int64,
    o2: tl.int64,
    o3: tl.int64,
    S: tl.int64,
    num_col_blocks,
    num_tiles,
    BR: tl.constexpr,
    BC: tl.constexpr,
    EVEN: tl.constexpr,
    GRID: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    if pid_y == 0:
        src = In0
    elif pid_y == 1:
        src = In1
    elif pid_y == 2:
        src = In2
    else:
        src = In3
    c = tl.where(pid_y == 0, c0, tl.where(pid_y == 1, c1, tl.where(pid_y == 2, c2, c3)))
    o = tl.where(pid_y == 0, o0, tl.where(pid_y == 1, o1, tl.where(pid_y == 2, o2, o3)))
    for t in tl.range(pid_x, num_tiles, GRID, num_stages=2):
        rb = t // num_col_blocks
        cb = t % num_col_blocks
        row0 = tl.cast(rb * BR, tl.int64)
        cols = cb * BC + tl.arange(0, BC)
        src0 = src + row0 * c
        dst0 = Out + row0 * S + o
        if EVEN:
            v0 = tl.load(src0 + cols)
            v1 = tl.load(src0 + tl.cast(1, tl.int64) * c + cols) if BR > 1 else v0
            v2 = tl.load(src0 + tl.cast(2, tl.int64) * c + cols) if BR > 2 else v0
            v3 = tl.load(src0 + tl.cast(3, tl.int64) * c + cols) if BR > 3 else v0
            tl.store(dst0 + cols, v0)
            if BR > 1:
                tl.store(dst0 + tl.cast(1, tl.int64) * S + cols, v1)
            if BR > 2:
                tl.store(dst0 + tl.cast(2, tl.int64) * S + cols, v2)
            if BR > 3:
                tl.store(dst0 + tl.cast(3, tl.int64) * S + cols, v3)
        else:
            m0 = (cols < c) & ((rb * BR + 0) < R)
            v0 = tl.load(src0 + cols, mask=m0, other=0)
            m1 = (cols < c) & ((rb * BR + 1) < R)
            v1 = (
                tl.load(src0 + tl.cast(1, tl.int64) * c + cols, mask=m1, other=0)
                if BR > 1
                else v0
            )
            m2 = (cols < c) & ((rb * BR + 2) < R)
            v2 = (
                tl.load(src0 + tl.cast(2, tl.int64) * c + cols, mask=m2, other=0)
                if BR > 2
                else v0
            )
            m3 = (cols < c) & ((rb * BR + 3) < R)
            v3 = (
                tl.load(src0 + tl.cast(3, tl.int64) * c + cols, mask=m3, other=0)
                if BR > 3
                else v0
            )
            tl.store(dst0 + cols, v0, mask=m0)
            if BR > 1:
                tl.store(dst0 + tl.cast(1, tl.int64) * S + cols, v1, mask=m1)
            if BR > 2:
                tl.store(dst0 + tl.cast(2, tl.int64) * S + cols, v2, mask=m2)
            if BR > 3:
                tl.store(dst0 + tl.cast(3, tl.int64) * S + cols, v3, mask=m3)


@triton.jit
def _rowvec_cat_kernel(
    Out,
    In0,
    In1,
    In2,
    In3,
    R,
    sp0,
    sp1,
    sp2,
    sp3,
    o0: tl.int64,
    o1: tl.int64,
    o2: tl.int64,
    o3: tl.int64,
    S: tl.int64,
    num_col_blocks,
    num_tiles,
    BR: tl.constexpr,
    BC: tl.constexpr,
    GRID: tl.constexpr,
):
    """Row-vectorized cat: a tile is shaped [BR, BC], so the row axis covers BR rows at once.

    Compared with `_blocked_cat_kernel` (which unrolls at most 4 rows), the wide BR
    folds the per-row copies of a narrow span into a single tile, cutting both the
    tile count and the scalar index overhead. It is enabled only for BC <= 32
    (narrow row spans); wide spans still use the unrolled kernel.
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    if pid_y == 0:
        src = In0
    elif pid_y == 1:
        src = In1
    elif pid_y == 2:
        src = In2
    else:
        src = In3
    sp = tl.where(
        pid_y == 0, sp0, tl.where(pid_y == 1, sp1, tl.where(pid_y == 2, sp2, sp3))
    )
    o = tl.where(pid_y == 0, o0, tl.where(pid_y == 1, o1, tl.where(pid_y == 2, o2, o3)))
    rows = tl.arange(0, BR)
    cols = tl.arange(0, BC)
    for t in tl.range(pid_x, num_tiles, GRID, num_stages=2):
        rb = t // num_col_blocks
        cb = t % num_col_blocks
        r_idx = tl.cast(rb * BR + rows, tl.int64)
        c_idx = cb * BC + cols
        tl.store(
            Out + r_idx[:, None] * S + (o + c_idx)[None, :],
            tl.load(src + r_idx[:, None] * sp + c_idx[None, :]),
        )


def _next_pow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pow2_floor(n):
    if n <= 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _can_fast_path(A, dim):
    """Legacy dim=0 boundary predicate (reused by row_stack / vstack__opt with the old semantics)."""
    if not (1 <= len(A) <= 4):
        return False
    rank = A[0].ndim
    d = dim + rank if dim < 0 else dim
    if d != 0 or rank < 1:
        return False
    first = A[0]
    if str(first.dtype) == "torch.float8_e8m0fnu":
        return False
    tail = tuple(first.shape[1:])
    for t in A:
        if (
            t.dtype != first.dtype
            or t.device != first.device
            or not t.is_contiguous()
            or t.ndim != rank
            or tuple(t.shape[1:]) != tail
        ):
            return False
    return True


def _is_fast_dtype(t):
    dt = t.dtype
    if dt.is_complex:
        return False
    if str(dt) == "torch.float8_e8m0fnu":
        return False
    return t.element_size() in (1, 2, 4)


def _int32_viewable(t):
    esize = t.element_size()
    if (t.numel() * esize) % 4 != 0:
        return False
    return (t.storage_offset() * esize) % 4 == 0


def _common_gate(A, dim):
    n = len(A)
    if not (2 <= n <= 4):
        return None
    first = A[0]
    rank = first.ndim
    if rank < 1:
        return None
    d = dim + rank if dim < 0 else dim
    if not (0 <= d < rank):
        return None
    if not _is_fast_dtype(first):
        return None
    pre = tuple(first.shape[:d])
    post = tuple(first.shape[d + 1 :])
    for t in A:
        if (
            t.dtype != first.dtype
            or t.device != first.device
            or not t.is_contiguous()
            or t.ndim != rank
            or tuple(t.shape[:d]) != pre
            or tuple(t.shape[d + 1 :]) != post
        ):
            return None
    return n, first, d, rank


def _make_plan(A, dim, out=None):
    gate = _common_gate(A, dim)
    if gate is None:
        return None
    n, first, d, rank = gate
    esize = first.element_size()
    out_shape = list(first.shape)
    out_shape[d] = sum(t.shape[d] for t in A)
    if out is not None:
        if (
            out.dtype != first.dtype
            or out.device != first.device
            or not out.is_contiguous()
            or tuple(out.shape) != tuple(out_shape)
        ):
            return None
        if any(out.data_ptr() == t.data_ptr() for t in A):
            return None

    if d == 0:
        word_size = 4
        if not all(_int32_viewable(t) for t in A):
            word_size = 1
        if out is not None and word_size == 4 and not _int32_viewable(out):
            word_size = 1
        ns = [t.numel() * esize // word_size for t in A]
        if max(ns) == 0:
            return {
                "empty": True,
                "out_shape": out_shape,
                "dtype": first.dtype,
                "device": first.device,
            }
        if max(ns) > _MAX_WORDS:
            return None
        even = len(set(ns)) == 1 and max(ns) % _BLOCK_EVEN == 0
        block = _BLOCK_EVEN if even else _BLOCK_MASKED
        plan = {
            "mode": "linear",
            "n": n,
            "dtype": first.dtype,
            "device": first.device,
            "out_shape": out_shape,
            "word_size": word_size,
            "ns": ns,
            "even": even,
            "BLOCK": block,
            "num_tiles": triton.cdiv(max(ns), block),
        }
    else:
        post_elems = 1
        for s in first.shape[d + 1 :]:
            post_elems *= s
        rows = 1
        for s in first.shape[:d]:
            rows *= s
        span_bytes = [t.shape[d] * post_elems * esize for t in A]
        word_size = 4
        if not all(_int32_viewable(t) for t in A) or any(
            sb % 4 != 0 for sb in span_bytes
        ):
            word_size = 1
        if out is not None and word_size == 4 and not _int32_viewable(out):
            word_size = 1
        spans = [sb // word_size for sb in span_bytes]
        if rows == 0 or max(spans) == 0:
            return {
                "empty": True,
                "out_shape": out_shape,
                "dtype": first.dtype,
                "device": first.device,
            }
        if rows > _MAX_WORDS or max(spans) > _MAX_WORDS:
            return None
        span_max = max(spans)
        block_c = _BLOCK_EVEN if span_max >= _BLOCK_EVEN else _next_pow2(span_max)
        block_r = min(4, _pow2_floor(rows), max(1, _TILE_TARGET // block_c))
        even = len(set(spans)) == 1 and span_max % block_c == 0 and rows % block_r == 0
        if not even:
            block_c = min(block_c, _BLOCK_MASKED)
            block_r = min(4, _pow2_floor(rows), max(1, _TILE_TARGET // block_c))
        plan = {
            "mode": "blocked",
            "n": n,
            "dtype": first.dtype,
            "device": first.device,
            "out_shape": out_shape,
            "word_size": word_size,
            "spans": spans,
            "rows": rows,
            "even": even,
            "BR": block_r,
            "BC": block_c,
            "S": sum(spans),
            "num_col_blocks": triton.cdiv(span_max, block_c),
            "num_tiles": triton.cdiv(rows, block_r) * triton.cdiv(span_max, block_c),
        }
        # Narrow row spans (e.g. hstack of [N, 2] / [N, 32] along dim=1): an unrolled
        # BR<=4 tile holds only 8~128 elements, so the tile count is high and the scalar
        # index overhead dominates. A row-vectorized tile (BR up to 32, at most 2048
        # words per tile) measured 1.13x ~ 1.46x.
        if len(set(spans)) == 1 and block_c <= 32 and rows >= 32:
            vec_r = min(32, _pow2_floor(rows))
            while vec_r > 1 and (rows % vec_r or vec_r * block_c > 2048):
                vec_r //= 2
            if rows % vec_r == 0 and spans[0] % block_c == 0:
                plan["mode"] = "rowvec"
                plan["BR"] = vec_r
                plan["num_tiles"] = triton.cdiv(rows, vec_r) * triton.cdiv(
                    span_max, block_c
                )
    offs = []
    acc = 0
    for size in plan["ns"] if plan["mode"] == "linear" else plan["spans"]:
        offs.append(acc)
        acc += size
    plan["offs"] = offs
    return plan


def _view_words(t, word_size):
    if word_size == 4:
        return t.reshape(-1).view(torch.int32)
    return t.reshape(-1).view(torch.uint8)


def _launch(plan, out_view, in_views):
    """Launch the plan's kernel through the cached raw launcher.

    The cat kernels take 15+ arguments; going through ``kernel[grid](...)``
    re-binds all of them on every call, which cost ~40us while the copy itself
    runs in a few microseconds. ``fast_launch`` caches the specialised artifact
    per argument shape and calls ``CompiledKernel.run`` directly (~5.6us).
    """
    n = plan["n"]
    ptrs = [in_views[j] if j < n else in_views[0] for j in range(4)]
    grid = (min(_GRID_CAP, plan["num_tiles"]), n)
    tensors = {
        "Out": out_view,
        "In0": ptrs[0],
        "In1": ptrs[1],
        "In2": ptrs[2],
        "In3": ptrs[3],
    }
    if plan["mode"] == "linear":
        scalars = {
            "n0": plan["ns"][0] if n > 0 else 0,
            "n1": plan["ns"][1] if n > 1 else 0,
            "n2": plan["ns"][2] if n > 2 else 0,
            "n3": plan["ns"][3] if n > 3 else 0,
            "o0": plan["offs"][0] if n > 0 else 0,
            "o1": plan["offs"][1] if n > 1 else 0,
            "o2": plan["offs"][2] if n > 2 else 0,
            "o3": plan["offs"][3] if n > 3 else 0,
            "num_tiles": plan["num_tiles"],
        }
        fast_launch.launch(
            _flat_cat_kernel,
            grid,
            tensors,
            scalars,
            {"BLOCK": plan["BLOCK"], "EVEN": plan["even"], "GRID": _GRID_CAP},
            num_warps=4,
            device=out_view.device,
        )
        return

    spans = [plan["spans"][j] if j < n else 0 for j in range(4)]
    offs = [plan["offs"][j] if j < n else 0 for j in range(4)]
    if plan["mode"] == "rowvec":
        scalars = {
            "R": plan["rows"],
            "sp0": spans[0],
            "sp1": spans[1],
            "sp2": spans[2],
            "sp3": spans[3],
            "o0": offs[0],
            "o1": offs[1],
            "o2": offs[2],
            "o3": offs[3],
            "S": plan["S"],
            "num_col_blocks": plan["num_col_blocks"],
            "num_tiles": plan["num_tiles"],
        }
        fast_launch.launch(
            _rowvec_cat_kernel,
            grid,
            tensors,
            scalars,
            {
                "BR": plan["BR"],
                "BC": plan["BC"],
                "GRID": min(_GRID_CAP, plan["num_tiles"]),
            },
            num_warps=4,
            device=out_view.device,
        )
        return

    scalars = {
        "R": plan["rows"],
        "c0": spans[0],
        "c1": spans[1],
        "c2": spans[2],
        "c3": spans[3],
        "o0": offs[0],
        "o1": offs[1],
        "o2": offs[2],
        "o3": offs[3],
        "S": plan["S"],
        "num_col_blocks": plan["num_col_blocks"],
        "num_tiles": plan["num_tiles"],
    }
    fast_launch.launch(
        _blocked_cat_kernel,
        grid,
        tensors,
        scalars,
        {"BR": plan["BR"], "BC": plan["BC"], "EVEN": plan["even"], "GRID": _GRID_CAP},
        num_warps=4,
        device=out_view.device,
    )


def try_fast_cat(A, dim):
    """Returns the concatenated result, or None when the fast path does not apply."""
    plan = _make_plan(A, dim)
    if plan is None:
        return None
    out = torch.empty(plan["out_shape"], dtype=plan["dtype"], device=plan["device"])
    if plan.get("empty"):
        return out
    try:
        out_view = _view_words(out, plan["word_size"])
        in_views = [_view_words(t, plan["word_size"]) for t in A]
    except (RuntimeError, TypeError, NotImplementedError):
        return None
    _launch(plan, out_view, in_views)
    return out


def try_fast_cat_out(A, dim, out):
    """cat.out fast path: writes straight into the given out, returning False when not applicable.

    Non-zero dims now also use the tiled channel; boundary cases (>4 inputs / float8 / mismatched
    dtype / non-contiguous input / non-contiguous out / shape or dtype mismatch / aliasing with an
    input / word count outside the i32 safe range) fall back to the full logic in `cat.py::cat_out`.
    """
    plan = _make_plan(A, dim, out=out)
    if plan is None:
        return False
    if plan.get("empty"):
        return True
    try:
        out_view = _view_words(out, plan["word_size"])
        in_views = [_view_words(t, plan["word_size"]) for t in A]
    except (RuntimeError, TypeError, NotImplementedError):
        return False
    _launch(plan, out_view, in_views)
    return True
