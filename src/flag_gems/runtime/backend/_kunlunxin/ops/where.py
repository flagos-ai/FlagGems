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
from _kunlunxin.utils.bf16_fast_store import bf16_fast_store
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from triton.runtime import driver

from flag_gems.ops.where import where_self_out as _generic_where_self_out
from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# WARNING: do NOT use a non-power-of-two unroll_num here. unroll_num=12 emits a
# kernel that issues a garbage AXI read address and wedges the device (dmesg:
# KL_XID64_AXI_ADDR_ERROR / KL_XID_KERNEL_EXCEPTION, then
# kl3_wait_for_noc_idle() timeout), requiring a soft_reset to recover. This was
# reproduced twice on a healthy card, so treat it as a compiler-side
# constraint, not a transient hardware fault. 8/16/32 are all safe.
#
# masked_fill-style config: isCloseVectorization keeps the mixed i1-mask
# tl.where on the fast path on XPU (masked_fill reaches 0.56x fp32 / 0.36x
# fp16 on 4096x4096, 2026-09-04). without it the bare pointwise_dynamic
# produces discrete access -> catastrophic latency (where_self dtype-balanced
# speedup 0.19 on the acceptance shape set, 2026-09-04 baseline). The
# sub/less_equal_ recipe (isCloseVectorization off) scalarizes the whole where
# kernel to 0.35x/0.19x because the bool CONDITION input is the bottleneck --
# not the tl.where vselect itself (an arithmetic rewrite a*c+b*(1-c) gives the
# same 0.35x; an int8 view of the condition plus sitofp hits a TritonXPU
# vector-widen lowering bug). isCloseVectorization is the only lever that helps.

# 2026-11-04: the float path used to be an arithmetic blend
# (`self * c + other * (1 - c)` over a condition widened from an int8 view, under
# a separate `config_openvec_` config that has since been removed). That rewrite
# is only equivalent to a select when both operands are finite: `nan * 0` and
# `inf * 0` are `nan`, so a nan/inf in the branch the condition does *not* select
# poisoned the result -- `where([True, False], [1, nan], [nan, 2])` returned
# `[nan, nan]` instead of `[1, 2]`, and `torch.logdet` (an `aten::where`
# composite) returned nan for every finite input. It also stopped compiling on
# the current TritonXPU toolchain for every numel >= 4096 (`size mismatch when
# packing elements for LLVM struct expected 4 but got 1` inside
# ConvertTritonXPUToLLVM), so the blend was both wrong and unusable at realistic
# sizes.
#
# Both paths now use a real `tl.where` select on a real bool condition.
# Re-measured on the 4096^2 fp32 case the old 2.9x result was taken on: the
# i1-condition select takes 0.2793 ms, while the int8-view select run under the
# retired `config_openvec_` takes 0.4204 ms -- 33% slower. The 2026-09-14 note
# above ("blend 131.9us vs i1 383.5us = 2.9x") no longer holds because the i1
# scalar-layout lowering it worked around has since been fixed; keeping i1 out of
# the dataflow no longer buys anything, and the blend itself does not compile at
# all from numel 4096 up.
config_closevec_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(1, 2, "NO_OPMATH")],
    config=config_closevec_,
)
@triton.jit
def where_inner(condition, self, other):
    return tl.where(condition, self, other)


# =============================================================================
# Flat-launcher fast path (same shape/contiguity/float cases, small numel).
#
# At [64,64] this operator is a 4096-element kernel whose device time is a
# fraction of a microsecond, so the call is entirely host-bound, and what it is
# bound on is `fn[grid](...)`: that re-specializes every argument, hashes a cache
# key, checks the used globals, then walks the operands again to drop constexprs
# and expand descriptors. Measured on node97 (evidence:
# artifacts/op-perf-batch-2026-09/evidence/where-self-node97-20260916/): 26.47us
# through the generic path against 5.41us for the *same compiled kernel* replayed
# through `driver.active.flat_launchers`. What this path adds is the kernel:
# `pointwise_dynamic` builds a general strided/broadcast kernel per rank and
# chooses tile size and cta count from the shape, all of which a 1-D,
# three-same-shape-plane case does not need.
# =============================================================================
_FLAT_MISS = object()
_FLAT_LAUNCHERS = _FLAT_MISS

# Above this many elements the generic path's tile/cta heuristic is worth more
# than the launcher overhead it costs, so the fast path declines.
_FLAT_MAX_NUMEL = 65536
# One tile per program; at 4096 the [64,64] case is a single program, exactly the
# tile size the generated wrapper picks for it today.
_FLAT_BLOCK = 4096


def _flat_launchers():
    """`driver.active.flat_launchers`, resolved once -- `driver.active` is a lazy
    proxy, so the attribute walk is not free at a ~6us a launch."""
    global _FLAT_LAUNCHERS
    if _FLAT_LAUNCHERS is _FLAT_MISS:
        _FLAT_LAUNCHERS = getattr(driver.active, "flat_launchers", None)
    return _FLAT_LAUNCHERS


@triton.jit(
    # Both of these exist only to keep the compiled kernel independent of values
    # the key below does not carry: without them, a pointer's alignment class or
    # `numel`'s divisibility would send the first call through a different
    # specialization than the one a bound launcher was built from.
    do_not_specialize=["numel"],
    do_not_specialize_on_alignment=["cond_ptr", "a_ptr", "b_ptr", "out_ptr"],
)
def _where_self_flat_kernel(
    cond_ptr,
    a_ptr,
    b_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    """`out = cond ? x : y` over three same-shape contiguous planes.

    Same math as `where_inner`: a real select, not an arithmetic blend, so a
    nan/inf in the branch the condition does not select is not propagated (see
    the 2026-11-04 note above). The condition arrives as an int8 view and is
    compared against zero inside the kernel. `numel` is a runtime bound and the
    store is masked, so a numel that BLOCK does not divide is served by the same
    binary.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    cond = tl.load(cond_ptr + offs, mask=mask)
    x = tl.load(a_ptr + offs, mask=mask)
    y = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.where(cond != 0, x, y), mask=mask)


def _flat_where_self(c, a, b, out, out_provided):
    """Launch `_where_self_flat_kernel` for the cases it covers; None otherwise.

    The launcher key is the caller's promise that the bound kernel serves the
    call. It is `(numel, a.dtype, b.dtype, out.dtype, BLOCK)`: `numel` and BLOCK
    fix the grid (which participates in compilation on XPU), the dtypes fix the
    compiled signature and the loads/stores, and BLOCK fixes the constexpr. It is
    complete because that is the whole kernel -- one flat extent, no strides, no
    broadcast, no descriptor -- and because the pointers are kept out of the
    compiled kernel entirely (`do_not_specialize_on_alignment`), which is the
    only reason they may be left out of the key.
    """
    if c.ndim == 0 or c.shape != a.shape or a.shape != b.shape:
        return None  # broadcast or scalar condition: the generic path has it
    if not (c.is_contiguous() and a.is_contiguous() and b.is_contiguous()):
        return None
    numel = a.numel()
    if numel == 0 or numel > _FLAT_MAX_NUMEL:
        return None
    # The launcher is bound per device, and the fast path launches on the current
    # one; the generated wrapper covers the general case with a device guard.
    if a.device.index != torch_device_fn.current_device():
        return None
    launchers = _flat_launchers()
    if launchers is None:  # triton without the launcher cache: correct, just slower
        return None
    if out_provided:
        if out.shape != a.shape or not out.is_contiguous():
            return None
    else:
        out = torch.empty(a.shape, dtype=a.dtype, device=a.device)

    key = (numel, a.dtype, b.dtype, out.dtype, _FLAT_BLOCK)
    grid = (-(-numel // _FLAT_BLOCK),)
    launch, stream = launchers.acquire(_where_self_flat_kernel, key)
    if launch is None:  # first call for this key: compile, bind, and replay next time
        kernel = _where_self_flat_kernel[grid](
            c.view(torch.int8), a, b, out, numel, BLOCK=_FLAT_BLOCK
        )
        launchers.bind(_where_self_flat_kernel, key, kernel, grid)
        return out
    launch(stream, c.view(torch.int8), a, b, out, numel)
    return out


def where_self_out(condition, self, other, out=None):
    logger.debug("GEMS_KUNLUNXIN WHERE_SELF_OUT")
    result_type = torch.result_type(self, other)
    if out is not None:
        assert (
            out.dtype == result_type
        ), f"Expected out type to be {result_type}, but got {out.dtype}."
    # bf16 outputs take the fast f32->bf16 store lowering for this launch only;
    # see _kunlunxin.utils.bf16_fast_store for why it is scoped rather than set
    # globally, and for the one-ULP tie-bias difference from the default path.
    with bf16_fast_store(result_type):
        return _where_self_out_impl(condition, self, other, out, result_type)


def _where_self_out_impl(condition, self, other, out, result_type):
    # `where_scalar_self` / `where_scalar_other` pass a Python number straight
    # through, so `self` or `other` may not be a tensor at all. The generic
    # implementation wraps non-tensors with `torch.tensor` before anything else;
    # this copy dropped that step and crashed with
    # `AttributeError: 'float' object has no attribute 'dtype'` on every scalar
    # case (72/72 in tests/test_where_self_out.py, on base too). The 0-dim
    # fixups below move the resulting CPU scalar onto the compute device.
    c, a, b = (
        x if isinstance(x, torch.Tensor) else torch.tensor(x)
        for x in (condition, self, other)
    )

    # Broadcasting operands go to the generic implementation. The pointwise
    # kernels generated for this backend do not cover every broadcast shape:
    # with a 0-dim operand against a one-element tensor, the generated wrapper
    # fails LLVM translation
    #
    #   error: LLVM Translation failed for operation:
    #     builtin.unrealized_conversion_cast
    #
    # for both the i1 and the int8-condition variants. The generic `tl.where`
    # kernel serves the same shapes correctly, and this branch is reached only
    # when the shapes differ -- the same-shape paths below are untouched.
    if c.shape != a.shape or a.shape != b.shape:
        return _generic_where_self_out(c, a, b, out=out)

    if a.dtype != result_type:
        a = a.to(result_type)
    if b.dtype != result_type:
        b = b.to(result_type)

    # Same validation as the original map/filter/set chain, but without
    # per-call lambdas + map/filter/set objects: on node97 the original form
    # measured +4.3us at [64,64] (official do_bench, 6-round same-window
    # bisect: base 9.15us -> +device-checks 14.54us).
    d_c, d_a, d_b = c.device, a.device, b.device
    if d_c.type != "cpu":
        device = d_c
    elif d_a.type != "cpu":
        device = d_a
    elif d_b.type != "cpu":
        device = d_b
    else:
        raise AssertionError("CPU only. There seems a mistake to dispatch to here.")
    for d in (d_c, d_a, d_b):
        if d.type != "cpu" and d != device:
            raise AssertionError(
                "Expected all tensors to be on the same device, but found at "
                f"least two devices, {[x.device for x in (c, a, b)]}"
            )
    if c.device != device and c.ndim == 0:
        c = torch.scalar_tensor(c.item(), dtype=c.dtype, device=device)
    if a.device != device and a.ndim == 0:
        a = torch.scalar_tensor(a.item(), dtype=a.dtype, device=device)
    if b.device != device and b.ndim == 0:
        b = torch.scalar_tensor(b.item(), dtype=b.dtype, device=device)

    assert (
        c.dtype == torch.bool
    ), f"where expected condition to be a boolean tensor, but got a tensor with dtype {condition.dtype}"

    out_provided = out is not None
    ndim = max(c.ndim, a.ndim, b.ndim)
    if result_type.is_floating_point:
        flat_out = _flat_where_self(c, a, b, out, out_provided)
        if flat_out is not None:
            return flat_out
        # Fallback for everything the fast path declines: broadcast shapes,
        # non-contiguous operands, numel over _FLAT_MAX_NUMEL, operands on a
        # device other than the current one, or a triton too old to have the
        # flat launcher cache.
        #
        # The condition is passed as a real bool tensor: `tl.where` takes an i1
        # predicate directly, and the int8-view workaround existed only to keep
        # i1 out of the arithmetic blend. Keeping it would make `tl.where` warn
        # ("non-boolean condition is deprecated") and, for 0-dim broadcast
        # operands, fail LLVM translation (`builtin.unrealized_conversion_cast`).
        #
        # 2026-09-16: when the caller did not pass `out`, let the generated
        # wrapper allocate the result instead of pre-allocating with
        # `torch.empty` and passing it as `out0=`.  Measured on node97,
        # [64,64] fp16, official do_bench: explicit empty + out0= 28.1us vs
        # wrapper self-allocation 6.4us (4.4x).  A pre-allocated *but reused*
        # out is also ~5.5us, so the cost sits in this per-call allocation
        # path, not in the kernel.  Semantics unchanged (both return a freshly
        # allocated tensor); the `out=` variant below is untouched.
        where_inner.instantiate(ndim)
        if out_provided:
            where_inner(c, a, b, out0=out)
        else:
            out = where_inner(c, a, b)
    else:
        if not out_provided:
            out_shape = torch.broadcast_shapes(c.shape, a.shape, b.shape)
            out = torch.empty(out_shape, dtype=result_type, device=device)
        where_inner.instantiate(ndim)
        where_inner(c, a, b, out0=out)
    return out


def where_self(condition, self, other):
    logger.debug("GEMS_KUNLUNXIN WHERE_SELF")
    return where_self_out(condition, self, other)


def where_scalar_self(condition, self, other):
    logger.debug("GEMS_KUNLUNXIN WHERE_SCALAR_SELF")
    return where_self_out(condition, self, other)


def where_scalar_other(condition, self, other):
    logger.debug("GEMS_KUNLUNXIN WHERE_SCALAR_OTHER")
    return where_self_out(condition, self, other)
