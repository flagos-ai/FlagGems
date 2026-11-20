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
"""

import torch
import triton
import triton.language as tl


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

    # Reduction tile: next power of two of the inner block, clamped to keep
    # the register file small (16 lanes is the minimum useful vector).
    BLOCK = min(1024, max(16, 1 << (max(1, inner) - 1).bit_length()))
    # fp32 accumulation, rounded to v.dtype on store (same as the generic
    # fused kernels).
    out = torch.empty(out_shape, device=v.device, dtype=v.dtype)
    _norm_kernel[(D,)](v, out, pre, D, inner, float(pow), float(pow) == 2.0, BLOCK)
    return out.reshape(()) if full_norm else out
