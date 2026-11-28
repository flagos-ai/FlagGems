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

"""Triton implementation of in-place elementwise multiply (aten::multiply_).

Semantics preserved:
  * ``other`` may be a Tensor (same shape or broadcastable to ``self``) or a
    Python scalar.
  * ``self`` is mutated in place and the *same* Python object is returned.
  * the promoted result dtype must be castable to ``self.dtype``.
  * when autograd is active the operation records an in-place multiply node
    (grad wrt the pre-multiply self is multiplied by ``other`` and vice versa),
    exactly like aten's ``Tensor.mul_``.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_INT32_MAX = 2**31 - 1

_DTYPE_NAMES = {
    torch.float32: "Float",
    torch.float64: "Double",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.int64: "Long",
    torch.int32: "Int",
    torch.int16: "Short",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.bool: "Bool",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}


def _dtype_name(dt):
    return _DTYPE_NAMES.get(dt, str(dt))


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #


@triton.jit
def _mul_flat_kernel(
    x_ptr,
    y_ptr,
    n,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    IS_BOOL: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        if IS_BOOL:
            tl.store(x_ptr + offs, x & y)
        else:
            tl.store(x_ptr + offs, x * y)
    else:
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        if IS_BOOL:
            tl.store(x_ptr + offs, x & y, mask=mask)
        else:
            tl.store(x_ptr + offs, x * y, mask=mask)


@triton.jit
def _mul_scalar_flat_kernel(
    x_ptr,
    alpha,
    n,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    IS_BOOL: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs)
        if IS_BOOL:
            tl.store(x_ptr + offs, x & (alpha != 0))
        else:
            tl.store(x_ptr + offs, x * alpha)
    else:
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        if IS_BOOL:
            tl.store(x_ptr + offs, x & (alpha != 0), mask=mask)
        else:
            tl.store(x_ptr + offs, x * alpha, mask=mask)


@triton.jit
def _mul_strided_kernel(
    x_ptr,
    y_ptr,
    alpha,
    n,
    SHAPE: tl.constexpr,
    XS: tl.constexpr,
    YS: tl.constexpr,
    NDIM: tl.constexpr,
    SCALAR: tl.constexpr,
    IDX64: tl.constexpr,
    IS_BOOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if IDX64:
        lin = offs.to(tl.int64)
        xo = tl.zeros((BLOCK,), dtype=tl.int64)
        yo = tl.zeros((BLOCK,), dtype=tl.int64)
    else:
        lin = offs
        xo = tl.zeros((BLOCK,), dtype=tl.int32)
        yo = tl.zeros((BLOCK,), dtype=tl.int32)
    for d in tl.static_range(NDIM - 1, -1, -1):
        c = lin % SHAPE[d]
        lin = lin // SHAPE[d]
        xo += c * XS[d]
        if not SCALAR:
            yo += c * YS[d]
    if SCALAR:
        x = tl.load(x_ptr + xo, mask=mask)
        if IS_BOOL:
            out = x & (alpha != 0)
        else:
            out = x * alpha
        tl.store(x_ptr + xo, out, mask=mask)
    else:
        x = tl.load(x_ptr + xo, mask=mask)
        y = tl.load(y_ptr + yo, mask=mask)
        if IS_BOOL:
            out = x & y
        else:
            out = x * y
        tl.store(x_ptr + xo, out, mask=mask)


# --------------------------------------------------------------------------- #
# Host helpers
# --------------------------------------------------------------------------- #


# Per-metadata launch plans. A key captures every observable property that
# selects a compiled specialization or a dispatch decision (shape, strides,
# dtypes, pointer alignment, scalar operand), so a repeat call with the same
# metadata reuses the same already-validated launcher. No tensor or tensor
# content is retained: only compiled handles and integer metadata.
_PLANS = {}

try:  # launch-side handles used to skip per-call JIT dispatch work
    from triton import knobs as _triton_knobs
    from triton.runtime import driver as _triton_driver
except Exception:  # pragma: no cover - fall back to the JIT launch path
    _triton_knobs = None
    _triton_driver = None


def _build_direct(kernel, grid3):
    """Bind the low-level launcher of an already compiled kernel.

    Returns ``direct(x, y, n, block, even, is_bool)``, or ``None`` when this
    Triton build does not expose the expected handles (the caller then keeps
    using the ordinary JIT launch path).
    """
    if _triton_knobs is None or _triton_driver is None:
        return None
    try:
        launcher = kernel.run
        function = kernel.function
        packed_metadata = kernel.packed_metadata
        launch_metadata = kernel.launch_metadata
        knobs_runtime = _triton_knobs.runtime
        driver_active = _triton_driver.active
    except Exception:
        return None
    gx, gy, gz = grid3
    raw_launch = getattr(launcher, "launch", None)
    cooperative = getattr(launcher, "launch_cooperative_grid", None)
    scratch_size = getattr(launcher, "profile_scratch_size", None)

    if raw_launch is not None and cooperative is not None and scratch_size == 0:
        # Inline version of HIPLauncher.__call__ for the common no-scratch case.
        def direct(x, y, n, block, even, is_bool):
            stream = driver_active.get_current_stream(
                driver_active.get_current_device()
            )
            metadata = launch_metadata(grid3, stream, x, y, n, block, even, is_bool)
            raw_launch(
                cooperative,
                gx,
                gy,
                gz,
                stream,
                function,
                None,
                packed_metadata,
                metadata,
                knobs_runtime.launch_enter_hook,
                knobs_runtime.launch_exit_hook,
                x,
                y,
                n,
                block,
                even,
                is_bool,
            )

        return direct

    def direct(x, y, n, block, even, is_bool):
        stream = driver_active.get_current_stream(driver_active.get_current_device())
        metadata = launch_metadata(grid3, stream, x, y, n, block, even, is_bool)
        launcher(
            gx,
            gy,
            gz,
            stream,
            function,
            packed_metadata,
            metadata,
            knobs_runtime.launch_enter_hook,
            knobs_runtime.launch_exit_hook,
            x,
            y,
            n,
            block,
            even,
            is_bool,
        )

    return direct


def _pick_block(n, element_size):
    if element_size == 4:
        block = 1024
    elif element_size <= 2:
        block = 2048
    else:
        block = 512
    if n < block:
        block = 32
        while block < n:
            block <<= 1
    return block


def _check_out_shape(ashape, oshape):
    """Raise the same errors aten raises for an invalid in-place broadcast."""
    try:
        out = torch.broadcast_shapes(ashape, oshape)
    except RuntimeError:
        na = len(ashape)
        nb = len(oshape)
        nd = max(na, nb)
        for i in range(nd):
            ia = i - (nd - na)
            ib = i - (nd - nb)
            sa = ashape[ia] if ia >= 0 else 1
            sb = oshape[ib] if ib >= 0 else 1
            if sa != sb and sa != 1 and sb != 1:
                raise RuntimeError(
                    "The size of tensor a (%d) must match the size of tensor b "
                    "(%d) at non-singleton dimension %d" % (sa, sb, i)
                ) from None
        raise
    if tuple(out) != tuple(ashape):
        raise RuntimeError(
            "output with shape %s doesn't match the broadcast shape %s"
            % (list(ashape), list(out))
        )


def _check_result_dtype(a, other, adt):
    rd = torch.result_type(a, other)
    if rd != adt:
        raise RuntimeError(
            "result type %s can't be cast to the desired output type %s"
            % (_dtype_name(rd), _dtype_name(adt))
        )


def _collapse(shape, xs, ys):
    """Collapse adjacent dimensions into an equivalent affine 1-D..n-D map."""
    dims = []
    for d in range(len(shape) - 1, -1, -1):
        s = shape[d]
        if s == 1:
            continue
        xd = xs[d]
        yd = ys[d]
        if dims:
            inner_s, inner_x, inner_y = dims[-1]
            if xd == inner_s * inner_x and yd == inner_s * inner_y:
                # Merged index c = outer_i * inner_s + inner_j must map to
                # c * inner_stride, so the merged stride is the INNER stride.
                dims[-1] = (s * inner_s, inner_x, inner_y)
                continue
        dims.append((s, xd, yd))
    dims.reverse()
    return dims


def _launch_strided(a, other, alpha, n):
    ashape = a.shape
    xs = a.stride()
    nd = len(ashape)
    if other is not None:
        oshape = other.shape
        ostr = other.stride()
        off = nd - len(oshape)
        ys = [0] * nd
        for d in range(len(oshape)):
            if oshape[d] != 1:
                ys[off + d] = ostr[d]
        ys = tuple(ys)
    else:
        ys = (0,) * nd

    dims = _collapse(ashape, xs, ys)
    if not dims:
        dims = [(1, 0, 0)]
    shape = tuple(d[0] for d in dims)
    xs_c = tuple(d[1] for d in dims)
    ys_c = tuple(d[2] for d in dims)
    nd_c = len(dims)

    max_off = 0
    for s, st in zip(shape, xs_c):
        max_off += (s - 1) * abs(st)
    for s, st in zip(shape, ys_c):
        max_off += (s - 1) * abs(st)
    idx64 = n > _INT32_MAX or max_off > _INT32_MAX

    block = _pick_block(n, a.element_size())
    grid = ((n + block - 1) // block,)
    is_scalar = other is None
    _mul_strided_kernel[grid](
        a,
        a if other is None else other,
        0.0 if alpha is None else alpha,
        n,
        SHAPE=shape,
        XS=xs_c,
        YS=ys_c,
        NDIM=nd_c,
        SCALAR=is_scalar,
        IDX64=idx64,
        IS_BOOL=(a.dtype is torch.bool),
        BLOCK=block,
        num_warps=4,
    )


def _noop(a, other):
    return


def _make_plan(a, other):
    """Validate, execute the first launch, and return a repeat-call launcher."""
    adt = a.dtype
    n = a.numel()
    is_bool = adt is torch.bool
    if isinstance(other, torch.Tensor):
        same_shape = other.shape == a.shape
        if not same_shape:
            _check_out_shape(a.shape, other.shape)
        if other.dtype is not adt:
            _check_result_dtype(a, other, adt)
        if n == 0:
            return _noop
        if (
            n <= _INT32_MAX
            and (same_shape or n == 1)
            and a.is_contiguous()
            and other.is_contiguous()
        ):
            block = _pick_block(n, a.element_size())
            # Three-stream (x, y, x) streaming: large tiles get more warps so
            # each SM keeps more loads in flight; tiny tensors get a narrower
            # tile so the multiply spreads over more CTAs at the same 16-byte
            # per-thread vector.
            if n >= 2097152:
                warps = 8
            else:
                warps = 4
                if n <= 8192 and block > 256:
                    block >>= 1
            even = n % block == 0
            gx = (n + block - 1) // block
            kernel = _mul_flat_kernel[(gx,)](
                a,
                other,
                n,
                BLOCK=block,
                EVEN=even,
                IS_BOOL=is_bool,
                num_warps=warps,
            )
            direct = _build_direct(kernel, (gx, 1, 1))
            if direct is not None:

                def launch(x, y, _d=direct, _n=n, _b=block, _e=even, _i=is_bool):
                    _d(x, y, _n, _b, _e, _i)

                return launch

            def launch(x, y, _n=n, _b=block, _e=even, _i=is_bool, _gx=gx, _w=warps):
                _mul_flat_kernel[(_gx,)](
                    x, y, _n, BLOCK=_b, EVEN=_e, IS_BOOL=_i, num_warps=_w
                )

            return launch
        _launch_strided(a, other, None, n)

        def launch(x, y, _n=n):
            _launch_strided(x, y, None, _n)

        return launch

    _check_result_dtype(a, other, adt)
    if n == 0:
        return _noop
    if n <= _INT32_MAX and a.is_contiguous():
        block = _pick_block(n, a.element_size())
        even = n % block == 0
        gx = (n + block - 1) // block
        kernel = _mul_scalar_flat_kernel[(gx,)](
            a,
            other,
            n,
            BLOCK=block,
            EVEN=even,
            IS_BOOL=is_bool,
            num_warps=4,
        )
        direct = _build_direct(kernel, (gx, 1, 1))
        if direct is not None:

            def launch(x, alpha, _d=direct, _n=n, _b=block, _e=even, _i=is_bool):
                _d(x, alpha, _n, _b, _e, _i)

            return launch

        def launch(x, alpha, _n=n, _b=block, _e=even, _i=is_bool, _gx=gx):
            _mul_scalar_flat_kernel[(_gx,)](
                x, alpha, _n, BLOCK=_b, EVEN=_e, IS_BOOL=_i, num_warps=4
            )

        return launch
    _launch_strided(a, None, other, n)

    def launch(x, alpha, _n=n):
        _launch_strided(x, None, alpha, _n)

    return launch


def _dispatch(a, other):
    """Launch the Triton kernel for ``a *= other`` (no autograd recording)."""
    if isinstance(other, torch.Tensor):
        key = (
            0,
            a.dtype,
            a.shape,
            a.stride(),
            a.data_ptr() & 15,
            other.dtype,
            other.shape,
            other.stride(),
            other.data_ptr() & 15,
        )
    else:
        key = (1, a.dtype, a.shape, a.stride(), a.data_ptr() & 15, type(other), other)
    plan = _PLANS.get(key)
    if plan is None:
        # First sight of this metadata: validate, launch, and bind the plan.
        _PLANS[key] = _make_plan(a, other)
    else:
        plan(a, other)


class _InplaceMul(torch.autograd.Function):
    """Autograd-recording wrapper for the in-place Triton multiply."""

    @staticmethod
    def forward(ctx, x, other):
        is_tensor = isinstance(other, torch.Tensor)
        ctx.is_tensor = is_tensor
        need_self = bool(is_tensor and ctx.needs_input_grad[1])
        ctx.need_self = need_self
        if need_self:
            ctx.save_for_backward(x.clone(), other)
        elif is_tensor:
            ctx.save_for_backward(other)
        else:
            ctx.scalar = other
        _dispatch(x, other)
        ctx.mark_dirty(x)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None
        if not ctx.is_tensor:
            gx = grad_out * ctx.scalar if ctx.needs_input_grad[0] else None
            return gx, None
        if ctx.need_self:
            saved_self, other = ctx.saved_tensors
        else:
            (other,) = ctx.saved_tensors
            saved_self = None
        gx = grad_out * other if ctx.needs_input_grad[0] else None
        go = (
            grad_out * saved_self
            if (ctx.need_self and ctx.needs_input_grad[1])
            else None
        )
        return gx, go


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def multiply_(self, other):
    logger.debug("GEMS_HYGON MULTIPLY_")
    a = self
    if not isinstance(a, torch.Tensor):
        raise TypeError("multiply_ expects the first argument to be a tensor")
    if torch.is_grad_enabled() and (
        a.requires_grad or (isinstance(other, torch.Tensor) and other.requires_grad)
    ):
        return _InplaceMul.apply(a, other)
    # Metadata validation (broadcast shape, result-dtype castability, empty
    # input) is performed once when the plan for this exact metadata is built;
    # the plan key contains every property those checks depend on.
    _dispatch(a, other)
    return a
