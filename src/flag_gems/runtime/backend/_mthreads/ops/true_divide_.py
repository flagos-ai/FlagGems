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

"""Triton implementation of aten::true_divide_.Tensor (in-place true division).

``run(self, other)`` divides ``self`` by ``other`` elementwise in place and
returns the same tensor object (same storage).  Semantics follow PyTorch:

* the division is computed in float32 opmath (as TensorIterator does for
  float16/bfloat16/float32) and cast back to ``self.dtype``,
* ``other`` may be broadcast against ``self.shape`` (trailing alignment),
  including 0-dim tensors and Python scalars,
* integer/bool ``self`` reproduces PyTorch's in-place cast error,
* the vendor dispatch-contract log line is emitted.

Layout: the hot path is a mask-free, statically addressed 1D kernel launched
through a cached compiled-kernel handle (bypassing the per-call Triton
specialization/cache-key machinery); masked / strided / scalar variants cover
the remaining layouts and are launched through the standard entry point.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _triton_driver

from flag_gems.ops.true_divide_ import true_divide_ as default_true_divide_
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_BLOCK = 1024
_WARPS = 4
_MAX_RANK = 8
_INT32_MAX = 0x7FFFFFFF
_DIV_LOG = "GEMS_MTHREADS TRUE_DIVIDE_"
_DEBUG = logging.DEBUG

_log = logging.getLogger(__name__)
# Loggers used by the vendor / canonical operator modules for the same contract.
_log_vendor = logging.getLogger("flag_gems.runtime.backend._mthreads.ops.div")
_log_canon = logging.getLogger("flag_gems.ops.true_divide_")

# Pre-built dispatch-contract record: emitting it through Logger.handle skips
# findCaller/record construction and all logger level checks, so the contract
# stays observable at ~1 us per call instead of ~25 us.
_DISPATCH_RECORD = logging.LogRecord(
    "flag_gems.runtime.backend._mthreads.ops.div",
    _DEBUG,
    __file__,
    0,
    _DIV_LOG,
    None,
    None,
    "true_divide_",
)
_ROOT_HANDLERS = logging.getLogger().handlers  # live list, updated by pytest

_DIMS_PAD = (1,) * _MAX_RANK
_STRIDE_PAD = (0,) * _MAX_RANK
_NO_SPEC = (
    ["n_elements", "val"]
    + [f"d{i}" for i in range(_MAX_RANK)]
    + [f"ss{i}" for i in range(_MAX_RANK)]
    + [f"os{i}" for i in range(_MAX_RANK)]
)


@triton.jit
def _div_flat_kernel(self_ptr, other_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(self_ptr + offs)
    b = tl.load(other_ptr + offs)
    tl.store(
        self_ptr + offs,
        (a.to(tl.float32) / b.to(tl.float32)).to(self_ptr.dtype.element_ty),
    )


@triton.jit(do_not_specialize=["n_elements"])
def _div_flat_masked_kernel(
    self_ptr,
    other_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_I64:
        offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        limit = n_elements.to(tl.int64)
    else:
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        limit = n_elements
    mask = offs < limit
    a = tl.load(self_ptr + offs, mask=mask)
    b = tl.load(other_ptr + offs, mask=mask)
    tl.store(
        self_ptr + offs,
        (a.to(tl.float32) / b.to(tl.float32)).to(self_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _div_bcast_kernel(self_ptr, other_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(self_ptr + offs)
    b = tl.load(other_ptr)
    tl.store(
        self_ptr + offs,
        (a.to(tl.float32) / b.to(tl.float32)).to(self_ptr.dtype.element_ty),
    )


@triton.jit(do_not_specialize=["n_elements"])
def _div_bcast_masked_kernel(
    self_ptr,
    other_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_I64:
        offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        limit = n_elements.to(tl.int64)
    else:
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        limit = n_elements
    mask = offs < limit
    a = tl.load(self_ptr + offs, mask=mask)
    b = tl.load(other_ptr)
    tl.store(
        self_ptr + offs,
        (a.to(tl.float32) / b.to(tl.float32)).to(self_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit(do_not_specialize=_NO_SPEC)
def _div_strided_kernel(
    self_ptr,
    other_ptr,
    n_elements,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    ss0,
    ss1,
    ss2,
    ss3,
    ss4,
    ss5,
    ss6,
    ss7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements.to(tl.int64)
    rem = offs
    s_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    o_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    if NDIM >= 1:
        i0 = rem % d0
        rem = rem // d0
        s_off += i0 * ss0.to(tl.int64)
        o_off += i0 * os0.to(tl.int64)
    if NDIM >= 2:
        i1 = rem % d1
        rem = rem // d1
        s_off += i1 * ss1.to(tl.int64)
        o_off += i1 * os1.to(tl.int64)
    if NDIM >= 3:
        i2 = rem % d2
        rem = rem // d2
        s_off += i2 * ss2.to(tl.int64)
        o_off += i2 * os2.to(tl.int64)
    if NDIM >= 4:
        i3 = rem % d3
        rem = rem // d3
        s_off += i3 * ss3.to(tl.int64)
        o_off += i3 * os3.to(tl.int64)
    if NDIM >= 5:
        i4 = rem % d4
        rem = rem // d4
        s_off += i4 * ss4.to(tl.int64)
        o_off += i4 * os4.to(tl.int64)
    if NDIM >= 6:
        i5 = rem % d5
        rem = rem // d5
        s_off += i5 * ss5.to(tl.int64)
        o_off += i5 * os5.to(tl.int64)
    if NDIM >= 7:
        i6 = rem % d6
        rem = rem // d6
        s_off += i6 * ss6.to(tl.int64)
        o_off += i6 * os6.to(tl.int64)
    if NDIM >= 8:
        i7 = rem % d7
        s_off += i7 * ss7.to(tl.int64)
        o_off += i7 * os7.to(tl.int64)
    a = tl.load(self_ptr + s_off, mask=mask)
    b = tl.load(other_ptr + o_off, mask=mask)
    tl.store(
        self_ptr + s_off,
        (a.to(tl.float32) / b.to(tl.float32)).to(self_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit(do_not_specialize=_NO_SPEC)
def _div_scalar_flat_kernel(
    self_ptr,
    val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_I64:
        offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        limit = n_elements.to(tl.int64)
    else:
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        limit = n_elements
    mask = offs < limit
    a = tl.load(self_ptr + offs, mask=mask)
    tl.store(
        self_ptr + offs,
        (a.to(tl.float32) / val).to(self_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit(do_not_specialize=_NO_SPEC)
def _div_scalar_strided_kernel(
    self_ptr,
    val,
    n_elements,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    ss0,
    ss1,
    ss2,
    ss3,
    ss4,
    ss5,
    ss6,
    ss7,
    NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements.to(tl.int64)
    rem = offs
    s_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    if NDIM >= 1:
        i0 = rem % d0
        rem = rem // d0
        s_off += i0 * ss0.to(tl.int64)
    if NDIM >= 2:
        i1 = rem % d1
        rem = rem // d1
        s_off += i1 * ss1.to(tl.int64)
    if NDIM >= 3:
        i2 = rem % d2
        rem = rem // d2
        s_off += i2 * ss2.to(tl.int64)
    if NDIM >= 4:
        i3 = rem % d3
        rem = rem // d3
        s_off += i3 * ss3.to(tl.int64)
    if NDIM >= 5:
        i4 = rem % d4
        rem = rem // d4
        s_off += i4 * ss4.to(tl.int64)
    if NDIM >= 6:
        i5 = rem % d5
        rem = rem // d5
        s_off += i5 * ss5.to(tl.int64)
    if NDIM >= 7:
        i6 = rem % d6
        rem = rem // d6
        s_off += i6 * ss6.to(tl.int64)
    if NDIM >= 8:
        i7 = rem % d7
        s_off += i7 * ss7.to(tl.int64)
    a = tl.load(self_ptr + s_off, mask=mask)
    tl.store(
        self_ptr + s_off,
        (a.to(tl.float32) / val).to(self_ptr.dtype.element_ty),
        mask=mask,
    )


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------

# Compiled-kernel handles for the mask-free contiguous path.  The key captures
# everything Triton specializes on (operand dtypes, pointer 16B alignment and
# the in-place aliasing case) plus the launch grid.
_FAST_RUNNERS = {}
_FAST_OK = True


def _build_flat_runner(self, other, grid0):
    """Compile once and return a bare-bones launch closure for this specialization.

    The closure keeps the resolved stream, kernel function and packed metadata,
    so a call only pays the device launcher invocation (no device/stream
    lookup, no launcher-closure creation, no launch-metadata construction, and
    no launcher wrapper when the kernel needs neither global scratch nor
    runtime argument expansion).
    """
    ck = _div_flat_kernel.warmup(self, other, _BLOCK, num_warps=_WARPS, grid=(1,))
    launcher = ck.run
    function = ck.function
    metadata = ck.packed_metadata
    stream = _triton_driver.active.get_current_stream(
        _triton_driver.active.get_current_device()
    )
    launch = getattr(launcher, "launch", None)
    needs_expand = getattr(launcher, "_needs_runtime_expansion", False)
    scratch = getattr(launcher, "global_scratch_size", None)
    block = _BLOCK
    if launch is not None and not needs_expand and not scratch:

        def runner(a, b):
            launch(
                grid0,
                1,
                1,
                stream,
                function,
                metadata,
                None,
                None,
                None,
                a,
                b,
                block,
            )

    else:

        def runner(a, b):
            launcher(
                grid0,
                1,
                1,
                stream,
                function,
                metadata,
                None,
                None,
                None,
                a,
                b,
                block,
            )

    return runner


def _disable_fast():
    global _FAST_OK
    _FAST_OK = False


def _collapse_self(self):
    """Fastest-varying-first (size, stride) pairs for ``self``, adjacent dims merged."""
    shape = self.shape
    stride = self.stride()
    dims = []
    strides = []
    for k in range(len(shape) - 1, -1, -1):
        d = shape[k]
        if d == 1:
            continue
        st = stride[k]
        if dims and st == strides[-1] * dims[-1]:
            dims[-1] *= d
        else:
            dims.append(d)
            strides.append(st)
    return dims, strides


def _collapse_both(self, other):
    """Aligned, collapsed (dims, self_strides, other_strides), fastest-first."""
    sshape = self.shape
    sstride = self.stride()
    oshape = other.shape
    ostride = other.stride()
    ndim = len(sshape)
    pad = ndim - len(oshape)
    dims = []
    ss = []
    os_ = []
    for k in range(ndim - 1, -1, -1):
        d = sshape[k]
        if d == 1:
            continue
        st = sstride[k]
        j = k - pad
        if j >= 0:
            ost = 0 if oshape[j] == 1 else ostride[j]
        else:
            ost = 0
        if dims and st == ss[-1] * dims[-1] and ost == os_[-1] * dims[-1]:
            dims[-1] *= d
        else:
            dims.append(d)
            ss.append(st)
            os_.append(ost)
    return dims, ss, os_


def _validate_broadcast(self, other):
    sshape = self.shape
    oshape = other.shape
    if len(oshape) > len(sshape):
        raise RuntimeError(
            "output with shape {} doesn't match the broadcast shape {}".format(
                list(sshape), list(oshape)
            )
        )
    pad = len(sshape) - len(oshape)
    for k in range(len(sshape) - 1, -1, -1):
        j = k - pad
        if j >= 0 and oshape[j] != 1 and oshape[j] != sshape[k]:
            raise RuntimeError(
                "output with shape {} doesn't match the broadcast shape {}".format(
                    list(sshape), list(oshape)
                )
            )


def _bad_dtype(self):
    if self.is_complex():
        raise RuntimeError("true_divide_: complex operands are not supported")
    raise RuntimeError(
        "result type Float can't be cast to the desired output type "
        + str(self.dtype).replace("torch.", "").capitalize()
    )


def _launch_flat(self, other, n):
    if (n & (_BLOCK - 1)) == 0 and n <= _INT32_MAX:
        _div_flat_kernel[(n // _BLOCK,)](
            self, other, BLOCK_SIZE=_BLOCK, num_warps=_WARPS
        )
    else:
        _div_flat_masked_kernel[((n + _BLOCK - 1) // _BLOCK,)](
            self,
            other,
            n,
            BLOCK_SIZE=_BLOCK,
            USE_I64=n > _INT32_MAX,
            num_warps=_WARPS,
        )


def _launch_bcast(self, other, n):
    if (n & (_BLOCK - 1)) == 0 and n <= _INT32_MAX:
        _div_bcast_kernel[(n // _BLOCK,)](
            self, other, BLOCK_SIZE=_BLOCK, num_warps=_WARPS
        )
    else:
        _div_bcast_masked_kernel[((n + _BLOCK - 1) // _BLOCK,)](
            self,
            other,
            n,
            BLOCK_SIZE=_BLOCK,
            USE_I64=n > _INT32_MAX,
            num_warps=_WARPS,
        )


def _launch_strided(self, other, n):
    dims, ss, os_ = _collapse_both(self, other)
    nd = len(dims)
    if nd > _MAX_RANK:
        raise RuntimeError("true_divide_: unsupported tensor rank")
    _div_strided_kernel[((n + _BLOCK - 1) // _BLOCK,)](
        self,
        other,
        n,
        *(tuple(dims) + _DIMS_PAD[: _MAX_RANK - nd]),
        *(tuple(ss) + _STRIDE_PAD[: _MAX_RANK - nd]),
        *(tuple(os_) + _STRIDE_PAD[: _MAX_RANK - nd]),
        NDIM=nd,
        BLOCK_SIZE=_BLOCK,
        num_warps=_WARPS,
    )


def _launch_scalar(self, val, n):
    if self.is_contiguous():
        _div_scalar_flat_kernel[((n + _BLOCK - 1) // _BLOCK,)](
            self,
            val,
            n,
            BLOCK_SIZE=_BLOCK,
            USE_I64=n > _INT32_MAX,
            num_warps=_WARPS,
        )
        return
    dims, ss = _collapse_self(self)
    nd = len(dims)
    if nd > _MAX_RANK:
        raise RuntimeError("true_divide_: unsupported tensor rank")
    _div_scalar_strided_kernel[((n + _BLOCK - 1) // _BLOCK,)](
        self,
        val,
        n,
        *(tuple(dims) + _DIMS_PAD[: _MAX_RANK - nd]),
        *(tuple(ss) + _STRIDE_PAD[: _MAX_RANK - nd]),
        NDIM=nd,
        BLOCK_SIZE=_BLOCK,
        num_warps=_WARPS,
    )


def _run_non_tensor(self, other):
    if not isinstance(self, torch.Tensor):
        raise TypeError("self must be a torch.Tensor")
    if not self.is_floating_point():
        _bad_dtype(self)
    if isinstance(other, complex):
        raise RuntimeError("true_divide_: complex operands are not supported")
    if self.numel() == 0:
        return self
    _launch_scalar(self, float(other), self.numel())
    return self


def _run_general(self, other):
    if not self.is_floating_point() or other.is_complex():
        _bad_dtype(self)
    if other.device != self.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least "
            "two devices"
        )
    n = self.numel()
    if other.shape != self.shape:
        _validate_broadcast(self, other)
        if n == 0:
            return self
        _launch_strided(self, other, n)
        return self
    if n == 0:
        return self
    if self.is_contiguous():
        if other.is_contiguous():
            _launch_flat(self, other, n)
            return self
        if other.numel() == 1:
            _launch_bcast(self, other, n)
            return self
    _launch_strided(self, other, n)
    return self


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def true_divide_(self, other):
    logger.debug("GEMS_MTHREADS TRUE_DIVIDE_")

    if not _use_triton_kernel(self) or (
        isinstance(other, torch.Tensor) and not _use_triton_kernel(other)
    ):
        return default_true_divide_(self, other)

    with torch_device_fn.device(self.device):
        return _launch_true_divide_(self, other)


def _launch_true_divide_(self, other):
    # dispatch-contract record (vendor message); ~1 us, no record construction
    if _ROOT_HANDLERS:
        _log.handle(_DISPATCH_RECORD)
        if _log_vendor.isEnabledFor(_DEBUG):
            _log_vendor.debug(_DIV_LOG)
        if _log_canon.isEnabledFor(_DEBUG):
            _log_canon.debug(_DIV_LOG)

    if not isinstance(other, torch.Tensor):
        return _run_non_tensor(self, other)
    if (
        self.is_contiguous()
        and other.is_contiguous()
        and other.shape == self.shape
        and self.is_floating_point()
        and other.device == self.device
    ):
        n = self.numel()
        if n == 0:
            return self
        if (n & (_BLOCK - 1)) == 0 and n <= _INT32_MAX and _FAST_OK:
            grid0 = n // _BLOCK
            key = (
                self.dtype,
                other.dtype,
                self is other,
                self.data_ptr() & 15 == 0,
                other.data_ptr() & 15 == 0,
                grid0,
            )
            runner = _FAST_RUNNERS.get(key)
            if runner is not None:
                runner(self, other)
                return self
            try:
                runner = _build_flat_runner(self, other, grid0)
            except Exception:
                _disable_fast()
                _launch_flat(self, other, n)
                return self
            _FAST_RUNNERS[key] = runner
            runner(self, other)
            return self
        _launch_flat(self, other, n)
        return self
    return _run_general(self, other)
