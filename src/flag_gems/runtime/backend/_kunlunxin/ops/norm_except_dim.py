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

"""Kunlunxin (XPU) override of ``norm_except_dim``.

The generic KernelGen implementation (``src/flag_gems/ops/norm_except_dim.py``)
fails on the XPU stack in two orthogonal ways:

* **``tl.atomic_add`` loses updates under multi-program contention.**  The
  generic split-reduction kernels (``_acc_outer_kernel`` / ``_acc_inner_kernel``)
  launch ``nrowchunks``/``nchunks`` programs that all ``atomic_add`` into the
  same output elements.  On this stack that RMW is not safe: with 264 programs
  contending on 17 addresses (``(8, 33, 17)``, ``dim=2``) the square-sum comes
  out at ~1/3 of the true value (``torch.norm_except_dim`` reference 14.3 vs
  4.8).  Verified with a slot-based variant (2-D grid + ``tl.store`` to a
  per-program buffer, then a host-side sum) that the same grid is fully
  launched and the per-program partials are exact -- only the ``atomic_add``
  itself is lossy.  (A 1-D grid with 264 programs atomically adding distinct
  rows reproduces the same loss, so it is the atomic op, not the launch.)
  The side effect is masked writes: the 256-lane vector atomic writes
  239 elements past a 17-element buffer, which also corrupts adjacent heap
  memory and is why later cases in the same process fail.

* **``tl.where(ax > 0.0, ...)`` cannot be selected for float32.**
  The generic ``pow != 2`` path compiles
  ``tl.where(ax > 0.0, tl.exp(POW * tl.log(ax)), 0.0)`` which on the XPU
  backend hits ``LLVM ERROR: Cannot select: v16i1 = setcc ... setone:ch``
  (``TritonXPULegalize``/backend selection).  Any float32 ``pow=1`` or
  ``pow=3`` case is uncompileable as a result.  float16 masks happen to be
  selectable, which is why only the fp32 rows fail.

The override below keeps one plan for every layout: **one program per output
element**, a single program reduces its whole ``(pre, inner)`` slab with a
fp32 accumulator and ``tl.store`` (no atomics anywhere), and the
``pow != 2`` path uses ``tl.maximum(ax, 1e-30)`` instead of a comparison
mask -- ``log``/``exp`` are plain FP ops on XPU and ``exp(POW * log(0))`` is
correctly 0, so clamping to the smallest f32 normal changes nothing while
staying selectable.  Numerical check on the target: max rel err 4.5e-4 (fp16),
1e-7 (fp32), ~2.5e-3 (bf16, output rounding) against the ATen reference.

The per-row pointer kernel is a cross-lane-reduction slow path on XPU3
(~30-60 GB/s: it was 0.20x vs torch on the official matrix).  For the common
contiguous ``dim == 0``, ``pow == 2`` layout it is therefore bypassed by the
``tle.gpu`` descriptor row-reduce below (same structure as the ``sum``
override's ``_tle_sum_row_kernel``, with the square folded into the loop):
a [XBLOCK, YBLOCK] LM tile is streamed through the cluster DMA and reduced
along axis 1, which reaches 0.8-1.5 TB/s.  Measured dtype-equal Gems Speedup
0.1961 -> ~1.4 on the official benchmark matrix; the pointer kernel stays as
the fallback for every other layout (other dims, ``pow != 2``, fp64), for
``inner < 64``, and whenever the TLE kernel cannot compile.
"""

import os

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle

    _HAS_TLE = True
except ImportError:
    _HAS_TLE = False

# The TLE cluster pipeline is opt-in on this stack; the `sum` override sets the
# same default for the same kernels.
os.environ.setdefault("TRITONXPU_TLE_PIPELINE", "1")

# LM budget of the TLE row tile: 4 KB per core on 64 cores (8 KB/core fails to
# allocate: "TLE kernel stack is over the local-memory budget").
_TLE_LM_BYTES_PER_CORE = 4096
_TLE_CORE_NUM = 64
# A row this wide gives each program fewer rows so the LM tile still fits and
# the transfers stay long (measured: xblock 256 -> 128 on (1024, 65536)).
_TLE_WIDE_ROW_BYTES = 32768
# Below this reduction length the [XBLOCK, YBLOCK] axis-1 reduce is not worth
# the LM round trip and the small-width reduce is the documented-miscompile
# zone; the pointer kernel keeps the case.
_TLE_MIN_INNER = 64

_TLE_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}
_TLE_ACC_DTYPE = {
    torch.float16: tl.float32,
    torch.float32: tl.float32,
    torch.bfloat16: tl.float32,
}


def _npo2(x):
    """Cheap next-power-of-two (triton.next_power_of_2 costs ~3us a call)."""
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _tle_available():
    """`tle.gpu` exists only on the xpu3 (KL3) cluster pipeline."""
    if not _HAS_TLE:
        return False
    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


_TLE_AVAILABLE = _tle_available()
# Latched when this toolchain cannot compile the TLE row kernel; every later
# call then takes the pointer kernel.  A failed compile leaves `out` untouched,
# so the retry is safe.
_TLE_BROKEN = False


@triton.jit(
    # Both descriptors come from freshly allocated tensors whose divisibility
    # classes vary; alignment must not enter the specialization key, and `N`
    # only bounds the loop.
    do_not_specialize=["N"],
    do_not_specialize_on_alignment=["a_desc", "c_desc"],
)
def _tle_norm2_row_kernel(
    a_desc,
    c_desc,
    N,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    NEED_ZERO: tl.constexpr,
):
    """``out[i] = sqrt(sum(a[i, :] ** 2))`` for the rows of a 2-D tile.

    ``tritonxpu-tle-core-tiling`` hands each core whole rows of the tile, so the
    axis-1 reduce stays core-local and needs no barrier.  The row is streamed in
    ``YBLOCK`` columns with a ``[XBLOCK]`` running accumulator.  A partial tile
    needs no mask: ``tle.gpu.copy`` clamps to the descriptor extents, so the only
    bad values are stale LM bytes, which ``NEED_ZERO`` clears on the one
    iteration that can be short.  The square is taken in the fp32 accumulator
    domain, so fp16/bf16 inputs reduce exactly like the fp32 pointer kernel.
    """
    pid = tl.program_id(0)
    row_off = pid * XBLOCK

    a_lmem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
    )
    c_lmem = tle.gpu.alloc([XBLOCK], dtype=OUT_DTYPE, layout=None, scope=tle.gpu.lmem)

    row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
    a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
    c_ptrs = tle.gpu.local_ptr(c_lmem, (tl.arange(0, XBLOCK),))

    acc = tl.zeros([XBLOCK], ACC_DTYPE)
    for coff in tl.range(0, N, YBLOCK):
        if NEED_ZERO:
            if coff + YBLOCK > N:
                tl.store(a_ptrs, tl.zeros([XBLOCK, YBLOCK], IN_DTYPE))
        tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
        v = tl.load(a_ptrs).to(ACC_DTYPE)
        acc = acc + tl.sum(v * v, axis=1)
    tl.store(c_ptrs, tl.sqrt(acc).to(OUT_DTYPE))
    tle.gpu.copy(c_lmem, c_desc, [XBLOCK], [row_off])


# (D, inner, itemsize) -> (xblock, yblock, grid, need_zero).
_TLE_GEOMS = {}


def _tle_geom(D, inner, itemsize):
    """Cached ``(xblock, yblock, grid, need_zero)`` for the squared row-reduce.

    Two effects set the geometry (measured on the official 5-shape x 3-dtype
    matrix): a program wants a full LM tile, and a grid of about one program per
    cluster.  Small ``D`` therefore takes XBLOCK=512 with the row padding clamped
    by the descriptors ((64, 64) is 5.7us at 512 against 14.0us at 64); at
    D = 1024 splitting into 4 programs of 256 rows beats both 128 (18.7us) and
    512 (15.1us); D = 4096 wants 512.  A wide row pulls XBLOCK back down so the
    LM tile still fits and the copies stay long.
    """
    key = (D, inner, itemsize)
    geom = _TLE_GEOMS.get(key)
    if geom is not None:
        return geom
    tile_elems = _TLE_LM_BYTES_PER_CORE * _TLE_CORE_NUM // itemsize
    xblock = min(512, max(256, _npo2(-(-D // 8))))
    if D <= 256:
        xblock = 512
    if inner * itemsize >= _TLE_WIDE_ROW_BYTES:
        xblock = min(xblock, 128)
    yblock = min(_npo2(inner), max(1, tile_elems // xblock))
    while yblock > inner and yblock > 1:
        yblock >>= 1
    geom = (xblock, yblock, -(-D // xblock), inner % yblock != 0)
    _TLE_GEOMS[key] = geom
    return geom


def _launch_tle_norm2(v2d, out1d):
    """One TLE row-reduce launch; returns False when it cannot be used."""
    global _TLE_BROKEN
    D, inner = v2d.shape
    xblock, yblock, grid, need_zero = _tle_geom(D, inner, v2d.element_size())
    try:
        from triton.tools.tensor_descriptor import TensorDescriptor

        _tle_norm2_row_kernel[(grid,)](
            TensorDescriptor.from_tensor(v2d, block_shape=[xblock, yblock]),
            TensorDescriptor.from_tensor(out1d, block_shape=[xblock]),
            inner,
            xblock,
            yblock,
            _TLE_TL_DTYPE[v2d.dtype],
            _TLE_ACC_DTYPE[v2d.dtype],
            _TLE_TL_DTYPE[v2d.dtype],
            need_zero,
        )
    except Exception:  # noqa: BLE001 - fall back to the pointer kernel
        _TLE_BROKEN = True
        return False
    return True


@triton.jit
def _norm_kernel(
    v_ptr,
    out_ptr,
    pre,
    D,
    inner,
    POW,
    IS_P2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Output element d = program_id(0); element v[p, d, q] lives at
    # p*D*inner + d*inner + q.  One program reduces the whole (pre, inner)
    # slab so no cross-program accumulation (atomic) is ever needed.
    d = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for p in range(0, pre):
        base = p * D * inner + d * inner
        for q0 in range(0, inner, BLOCK):
            q = q0 + tl.arange(0, BLOCK)
            mask = q < inner
            x = tl.load(v_ptr + base + q, mask=mask, other=0.0).to(tl.float32)
            if IS_P2:
                acc += x * x
            else:
                ax = tl.abs(x)
                # No `ax > 0.0` compare (unselectable on XPU f32); clamping to
                # the smallest f32 normal makes log finite and contributes
                # exactly 0 in f32 for every |x| below it, so exp(POW*log(*))
                # stays the p-th power of |x|.
                acc += tl.exp(POW * tl.log(tl.maximum(ax, 1e-30)))
    s = tl.sum(acc, axis=0)
    if IS_P2:
        r = tl.sqrt(s)
    else:
        r = tl.exp((1.0 / POW) * tl.log(s))
    tl.store(out_ptr + d, r)


def norm_except_dim(v, pow=2, dim=0):
    """``torch.norm_except_dim``-compatible p-norm over all dims but one.

    Same contract as the generic KernelGen implementation: keeps ``dim``,
    reduces the rest, ``dim == -1`` means the norm of the whole tensor (scalar
    output).  Returns a tensor of ``v.dtype``.
    """
    assert v.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"norm_except_dim: unsupported dtype {v.dtype}"

    v = v.contiguous()
    # aten treats dim == -1 as a sentinel meaning "norm over the whole tensor"
    # (0-dim scalar output), NOT "the last dim".  Only the literal -1 is
    # special; other negative dims index from the end as usual.
    full_norm = int(dim) == -1
    if full_norm:
        v = v.reshape(1, -1)
    ndim = v.dim()
    d = 0 if full_norm else int(dim) % ndim
    numel = v.numel()
    D = v.size(d)
    inner = 1
    for k in range(d + 1, ndim):
        inner *= v.size(k)
    pre = numel // (D * inner) if D * inner > 0 else 0

    out_shape = [1] * ndim
    out_shape[d] = D

    if numel == 0:
        acc = torch.zeros(out_shape, device=v.device, dtype=v.dtype)
        return acc.reshape(()) if full_norm else acc

    is_p2 = float(pow) == 2.0
    # Fast path: dim == 0 reduces the contiguous trailing block of each output
    # element, which is exactly the `tle.gpu` row-reduce layout.  Everything
    # else keeps the pointer kernel below.
    if (
        not full_norm
        and d == 0
        and is_p2
        and inner >= _TLE_MIN_INNER
        and v.dtype in _TLE_TL_DTYPE
        and _TLE_AVAILABLE
        and not _TLE_BROKEN
    ):
        acc = torch.empty(out_shape, device=v.device, dtype=v.dtype)
        if _launch_tle_norm2(v.view(D, inner), acc.view(D)):
            return acc
        # TLE compile failed: fall through to the pointer kernel (`acc` has not
        # been written).

    # Reduction tile: next power of two of the inner block, clamped to keep
    # the register file small (16 lanes is the minimum useful vector).
    BLOCK = min(1024, max(16, 1 << (max(1, inner) - 1).bit_length()))
    # fp32 accumulation, rounded to v.dtype on store (same as the generic
    # fused kernels).
    out = torch.empty(out_shape, device=v.device, dtype=v.dtype)
    _norm_kernel[(D,)](v, out, pre, D, inner, float(pow), is_p2, BLOCK)
    return out.reshape(()) if full_norm else out
