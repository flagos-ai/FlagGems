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

"""Wrap an integer tensor as a per-tensor affine quantized tensor.

``aten::_make_per_tensor_quantized_tensor(Tensor self, float scale, int
zero_point) -> Tensor`` derives the quantized dtype from the input dtype,
stores ``scale`` / ``zero_point`` verbatim, and copies the input values into
the ``int_repr``. Measured on the target build: ``uint8 -> quint8``,
``int8 -> qint8``, ``int32 -> qint32``; the result is a fresh allocation whose
``int_repr()`` compares equal to the input and does not share its storage.

The device work is the copy into the quantized tensor's raw storage: a flat
streaming kernel for contiguous inputs and a stride gather otherwise.

enflame GCU adaptation: without ``TORCH_GCU_ENABLE_INT64_AND_UINT64`` the
torch_gcu ATen layer silently stores device int64 tensors as packed int32
while the metadata still claims int64, so the gather kernel's 8-byte
shape/stride loads read interleaved garbage and the wild offsets abort the
device. The strided path therefore carries its shape/stride sidecars in the
int32 safe domain (every realistic tensor fits) and the kernel upcasts the
loaded scalars to int64 explicitly before the offset arithmetic; only shape
or stride values at or beyond 2**31 fall back to the int64 sidecars.

The gather loop is also unrolled to the input's real rank rather than the
``_MAXD`` floor shared with the common implementation. A fixed ``MAXD``
silently drops every dimension beyond it, so any non-contiguous input of rank
greater than ``MAXD`` wrote the leading slices on top of each other (measured
on GCU400: rank 9 lost 768 of 1536 elements, rank 10 lost 2304 of 3072).
"""

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_QDTYPE = {
    torch.uint8: torch.quint8,
    torch.int8: torch.qint8,
    torch.int32: torch.qint32,
}

# Lower bound for the gather loop unroll. Ranks up to this are handled by a
# fixed unroll so the common low-rank case compiles to the same code as the
# common implementation; taller ranks unroll to their real rank.
_MAXD = 8

_INT32_MAX = (1 << 31) - 1

_INT64_ENV = "TORCH_GCU_ENABLE_INT64_AND_UINT64"


def _int64_device_storage_is_trustworthy():
    """Whether device int64 tensors hold real int64 data on this backend.

    Without ``TORCH_GCU_ENABLE_INT64_AND_UINT64`` torch_gcu materialises
    device int64 tensors as packed int32 under int64 metadata.  A size-1
    sidecar happens to read back correctly because the padding upper half is
    zero, but a longer one reads interleaved garbage — so the fallback branch
    below cannot be used at all in that regime.
    """
    return os.environ.get(_INT64_ENV, "0").strip().lower() not in ("", "0", "false")


@triton.jit
def _copy_1d_kernel(src, dst, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(src + offs, mask=mask)
    tl.store(dst + offs, x, mask=mask)


@triton.jit
def _gather_1d_kernel(
    src,
    dst,
    n_elements,
    shape_ptr,
    stride_ptr,
    ndim,
    BLOCK: tl.constexpr,
    MAXD: tl.constexpr,
):
    # Copy the logical (row-major) values of a strided input into a contiguous
    # destination: decompose each flat index into per-dim coordinates and gather
    # through the input strides. The loop always runs MAXD steps and masks out
    # the dimensions the input does not have, so ndim stays a runtime value.
    # MAXD must be at least ndim: every step handles one dimension, so a
    # smaller MAXD silently ignores the leading coordinates and gathers the
    # wrong element.
    # The shape/stride sidecars may arrive as int32 (the torch_gcu int64-safe
    # domain, see the module docstring) or int64; the loaded scalars are
    # upcast explicitly so the offset arithmetic is always 64-bit regardless
    # of the sidecar width.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    off = tl.zeros((BLOCK,), dtype=tl.int64)
    rem = offs.to(tl.int64)
    for d in tl.static_range(MAXD):
        dd = ndim - 1 - d
        active = dd >= 0
        dd_safe = tl.maximum(dd, 0)
        s = tl.load(shape_ptr + dd_safe).to(tl.int64)
        st = tl.load(stride_ptr + dd_safe).to(tl.int64)
        s_safe = tl.where(active, s, 1)
        st_safe = tl.where(active, st, 0)
        c = rem % s_safe
        off += c * st_safe
        rem = rem // s_safe
    x = tl.load(src + off, mask=mask, other=0)
    tl.store(dst + offs, x, mask=mask)


def _copy_config(dtype, n):
    if dtype == torch.int32:
        return 1024, 8
    if n < (1 << 21):
        return 8192, 4
    return 16384, 8


def _index_sidecars(self):
    """Device sidecar tensors carrying ``self``'s shape and strides.

    torch_gcu without ``TORCH_GCU_ENABLE_INT64_AND_UINT64`` materialises
    device int64 tensors as packed int32 under int64 metadata (silent
    narrowing), which poisons the gather kernel's 8-byte loads. Shape and
    stride values sit deep inside the int32 domain for any realistic tensor,
    so the sidecars default to true int32 tensors — regime-independent on
    this backend. Only values at or beyond 2**31 (a strided dimension of
    2 GiB+ payload) fall back to int64 sidecars, which need the env on
    torch_gcu; the kernel upcasts either width explicitly. That fallback
    raises when the env is missing rather than corrupting the result.
    """
    shape = tuple(int(v) for v in self.shape)
    stride = tuple(int(v) for v in self.stride())
    if max(shape + stride, default=0) <= _INT32_MAX:
        dtype = torch.int32
    else:
        # A dimension spans 2**31 elements or more, so the sidecars cannot be
        # narrowed to int32.  The int64 sidecars are only sound when the
        # backend actually stores int64; otherwise the kernel reads packed
        # int32 garbage and either writes wrong data or aborts the device.
        # Fail loudly instead.
        if not _int64_device_storage_is_trustworthy():
            raise RuntimeError(
                "make_per_tensor_quantized_tensor: a strided input needs "
                "int64 shape/stride sidecars (max shape/stride "
                f"{max(shape + stride)} exceeds the int32 range), but "
                f"{_INT64_ENV} is not set. Without it torch_gcu stores "
                "device int64 tensors as packed int32, the gather kernel "
                "would read corrupt strides and either write wrong data or "
                f"abort the device. Set {_INT64_ENV}=1 to run this case."
            )
        dtype = torch.int64
    shape_t = torch.tensor(shape, device=self.device, dtype=dtype)
    stride_t = torch.tensor(stride, device=self.device, dtype=dtype)
    return shape_t, stride_t


def _make_per_tensor_quantized_tensor(
    self: torch.Tensor, scale: float, zero_point: int
) -> torch.Tensor:
    """Wrap integer values into a per-tensor affine quantized tensor.

    Args:
        self (torch.Tensor): Integer tensor with dtype uint8, int8 or int32.
        scale (float): Quantization scale, stored verbatim.
        zero_point (int): Quantization zero point, stored verbatim.

    Returns:
        torch.Tensor: Quantized tensor whose ``int_repr()`` equals ``self``.
    """
    logger.debug("GEMS_ENFLAME_GCU400 MAKE_PER_TENSOR_QUANTIZED_TENSOR")
    qdtype = _QDTYPE[self.dtype]
    out = torch._empty_affine_quantized(
        self.shape,
        dtype=qdtype,
        device=self.device,
        scale=float(scale),
        zero_point=int(zero_point),
    )
    n = self.numel()
    if n == 0:
        return out

    # Zero-copy view over the quantized tensor's raw storage, so the device
    # kernel writes the int representation that ``int_repr()`` exposes.
    raw = torch.empty(0, dtype=self.dtype, device=self.device).set_(
        out.untyped_storage(), 0, (n,)
    )
    if self.is_contiguous():
        block, num_warps = _copy_config(self.dtype, n)
        grid = (triton.cdiv(n, block),)
        _copy_1d_kernel[grid](self, raw, n, BLOCK=block, num_warps=num_warps)
    else:
        shape_t, stride_t = _index_sidecars(self)
        block = 1024
        grid = (triton.cdiv(n, block),)
        _gather_1d_kernel[grid](
            self,
            raw,
            n,
            shape_t,
            stride_t,
            self.dim(),
            BLOCK=block,
            # Unroll to the real rank: a fixed _MAXD drops any dimension past
            # it, collapsing the leading slices of rank > _MAXD inputs.
            MAXD=max(self.dim(), _MAXD),
            num_warps=8,
        )
    return out


def _make_per_tensor_quantized_tensor_out(
    self: torch.Tensor,
    scale: float,
    zero_point: int,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """Write the quantized wrap of ``self`` into ``out`` and return ``out``.

    Args:
        self (torch.Tensor): Integer tensor with dtype uint8, int8 or int32.
        scale (float): Quantization scale.
        zero_point (int): Quantization zero point.
        out (torch.Tensor): Destination quantized tensor.

    Returns:
        torch.Tensor: ``out``, holding the values of ``self``.
    """
    logger.debug("GEMS_ENFLAME_GCU400 MAKE_PER_TENSOR_QUANTIZED_TENSOR_OUT")
    n = self.numel()
    if n == 0:
        return out
    raw = torch.empty(0, dtype=self.dtype, device=self.device).set_(
        out.untyped_storage(), 0, (n,)
    )
    if self.is_contiguous():
        block, num_warps = _copy_config(self.dtype, n)
        grid = (triton.cdiv(n, block),)
        _copy_1d_kernel[grid](self, raw, n, BLOCK=block, num_warps=num_warps)
    else:
        shape_t, stride_t = _index_sidecars(self)
        block = 1024
        grid = (triton.cdiv(n, block),)
        _gather_1d_kernel[grid](
            self,
            raw,
            n,
            shape_t,
            stride_t,
            self.dim(),
            BLOCK=block,
            # Unroll to the real rank: a fixed _MAXD drops any dimension past
            # it, collapsing the leading slices of rank > _MAXD inputs.
            MAXD=max(self.dim(), _MAXD),
            num_warps=8,
        )
    return out
