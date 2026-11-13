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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# One program covers BLOCK complex elements (2 * BLOCK reals); a wide block
# keeps the grid small and lets Triton saturate HBM on the streaming copy
# while leaving enough programs for the launch-bound small cases.
_BLOCK_SIZE = 1024
# Leading (batch) dimensions are decomposed with a fixed number of slots so
# the kernel always receives its full argument sets; higher-rank inputs are
# rejected by the wrapper.
_MAX_LEAD_RANK = 5

_COMPLEX_DTYPE = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}
_COMPLEX_TO_REAL = {complex: real for real, complex in _COMPLEX_DTYPE.items()}


@triton.jit
def _view_as_complex_copy_contig_kernel(
    src_ptr,
    dst_ptr,
    n_complex,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy a contiguous ``(..., 2)`` real buffer into complex storage.

    ``src_ptr`` and ``dst_ptr`` point at the same number of reals: the input
    is a contiguous real tensor whose last dimension has size 2 and the
    destination is the real view of a contiguous complex tensor. Complex
    element ``i`` therefore occupies the adjacent real slots ``2 * i`` and
    ``2 * i + 1``, so one linear pass over the real stream is both exact and
    fully coalesced.
    """
    pid = tl.program_id(axis=0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i < n_complex
    i = i.to(tl.int64)

    lane = tl.arange(0, 2)
    offsets = i[:, None] * 2 + lane[None, :]
    mask2 = mask[:, None]
    values = tl.load(src_ptr + offsets, mask=mask2)
    tl.store(dst_ptr + offsets, values, mask=mask2)


@triton.jit
def _view_as_complex_copy_strided_kernel(
    src_ptr,
    dst_ptr,
    d0,
    d1,
    d2,
    d3,
    d4,
    s0,
    s1,
    s2,
    s3,
    s4,
    n_complex,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy a non-contiguous ``(..., 2)`` real buffer into complex storage.

    The output is contiguous, so the flat complex index decomposes row-major
    over the leading (batch) dimensions; each recovered coordinate is scaled
    by the matching *input* stride (expressed in real elements). The trailing
    dimension keeps stride 1, so the (real, imag) pair of a complex element is
    always adjacent and is loaded as a two-wide vector.

    The slots carry the leading dimensions in input order: ``d0``/``s0`` is
    the outermost lead dimension and ``d[RANK-1]``/``s[RANK-1]`` the innermost
    one (the dimension immediately before the complex pair). The generic path
    peels digits from the innermost slot outward, so both paths share the same
    convention.

    A ``RANK == 2`` specialization is kept because it is the overwhelmingly
    common non-contiguous case (a row/column slice of an otherwise flat
    buffer); the tight two-step ``div``/``sub`` decomposition beats the generic
    static loop by ~40% on that layout.
    """
    pid = tl.program_id(axis=0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i < n_complex
    i = i.to(tl.int64)

    if RANK == 2:
        # d1/s1 is the innermost lead dim (fastest digit), d0/s0 the outer one.
        outer = i // d1
        src_base = (i - outer * d1) * s1 + outer * s0
    else:
        lead_dims = [d0, d1, d2, d3, d4]
        lead_strides = [s0, s1, s2, s3, s4]
        rem = i
        src_base = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        for j in tl.static_range(RANK - 1, -1, -1):
            idx = rem % lead_dims[j]
            rem = rem // lead_dims[j]
            src_base += idx * lead_strides[j]

    lane = tl.arange(0, 2)
    src_offsets = src_base[:, None] + lane[None, :]
    dst_offsets = i[:, None] * 2 + lane[None, :]
    mask2 = mask[:, None]
    values = tl.load(src_ptr + src_offsets, mask=mask2)
    tl.store(dst_ptr + dst_offsets, values, mask=mask2)


def view_as_complex_copy(self: torch.Tensor) -> torch.Tensor:
    r"""Copy a real tensor whose last dimension has size 2 into a complex tensor.

    Mirrors ``aten::view_as_complex_copy``: the out-of-place counterpart of
    ``torch.view_as_complex``. The input's last dimension must have size 2
    and stride 1 (batch dimensions may be non-contiguous); ``float32`` maps
    to ``complex64``, ``float64`` to ``complex128`` and ``float16`` to
    ``complex32``. The result is a contiguous, freshly allocated tensor that
    never aliases the input. The same errors PyTorch raises are raised: 0-dim
    inputs, a wrong last-dimension size or stride, and unsupported dtypes
    (bfloat16 and every non-float type).
    """
    logger.debug("GEMS VIEW_AS_COMPLEX_COPY")

    if self.dim() == 0:
        raise RuntimeError("Input tensor must have one or more dimensions")
    if self.size(-1) != 2:
        raise RuntimeError("Tensor must have a last dimension of size 2")
    complex_dtype = _COMPLEX_DTYPE.get(self.dtype)
    if complex_dtype is None:
        raise RuntimeError(
            "view_as_complex is only supported for half, float and double "
            "tensors, but got a tensor of scalar type: "
            f"{str(self.dtype).replace('torch.', '').capitalize()}"
        )
    if self.stride(-1) != 1:
        raise RuntimeError("Tensor must have a last dimension with stride 1")

    # ``view_as_real`` refuses tensors carrying a lazy conjugate or negative
    # bit; resolving them first is an exact identity for the reinterpreted
    # values and keeps the copy on the logical data.
    if self.is_conj():
        self = self.resolve_conj()
    if self.is_neg():
        self = self.resolve_neg()

    leading = tuple(self.shape[:-1])
    out = torch.empty(leading, dtype=complex_dtype, device=self.device)
    n_complex = out.numel()
    if n_complex == 0:
        return out

    out_real = torch.view_as_real(out)
    with torch_device_fn.device(self.device):
        grid = lambda meta: (triton.cdiv(n_complex, meta["BLOCK_SIZE"]),)
        if self.is_contiguous():
            _view_as_complex_copy_contig_kernel[grid](
                self,
                out_real,
                n_complex,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
        else:
            rank = self.dim() - 1
            if rank > _MAX_LEAD_RANK:
                raise RuntimeError(
                    f"view_as_complex_copy supports at most {_MAX_LEAD_RANK + 1} "
                    f"dimensions, got {self.dim()}"
                )
            lead_dims = list(self.shape[:-1]) + [1] * (_MAX_LEAD_RANK - rank)
            lead_strides = list(self.stride()[:-1]) + [0] * (_MAX_LEAD_RANK - rank)
            _view_as_complex_copy_strided_kernel[grid](
                self,
                out_real,
                *lead_dims,
                *lead_strides,
                n_complex,
                RANK=rank,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
    return out


# ---------------------------------------------------------------------------
# torch.Tensor.to complex32-cast shim (torch_gcu P21 defect workaround)
#
# topsatenCopy aborts the whole process for any on-device dtype cast whose
# *source* is complex32 (ComplexHalf -> ComplexFloat/ComplexDouble hits
# TOPSATEN_STATUS_NOT_SUPPORT and SIGABRTs instead of raising), which kills
# every consumer that upcasts a complex32 gcu tensor -- e.g. the accuracy
# harness comparing float16->complex32 results through a lossless complex64
# upcast (complex32 has no CPU equal kernel).  ENFLAME_PT_OP_DEBUG_CONFIG
# cannot intercept this: _to_copy has a gcu kernel registration and never
# enters the fallback_cpu list (P21).  The unsupported cast decomposes into
# steps topsaten fully supports: view the complex tensor as its real dtype,
# widen the (real, imag) pair to the target's real dtype, and view back as
# complex -- bit-exact and on device.  Layouts whose view_as_real would be
# rejected fall back to a value-identical CPU round trip.  Every other call
# shape passes through to the original ``to`` untouched, and only
# complex32-on-gcu sources are intercepted (complex64/complex128 casts are
# natively supported).  Drop once topsatenCopy handles ComplexHalf.
# ---------------------------------------------------------------------------


def _install_complex_cast_to_shim():
    current = torch.Tensor.to
    if getattr(current, "_flaggems_gcu_complex_cast_safe", False):
        return

    def _complex_cast_safe_to(self, *args, **kwargs):
        if args:
            target_dtype = args[0] if isinstance(args[0], torch.dtype) else None
        elif kwargs.get("device") is None:
            target_dtype = kwargs.get("dtype")
        else:
            target_dtype = None
        if (
            target_dtype is not None
            and target_dtype != self.dtype
            and target_dtype in _COMPLEX_TO_REAL
            and self.dtype == torch.complex32
            and self.device.type == "gcu"
        ):
            try:
                real_pair = torch.view_as_real(self)
                real_pair = current(real_pair, _COMPLEX_TO_REAL[target_dtype])
                return torch.view_as_complex(real_pair)
            except RuntimeError:
                on_host = current(self, "cpu")
                return current(current(on_host, target_dtype), self.device)
        return current(self, *args, **kwargs)

    _complex_cast_safe_to._flaggems_gcu_complex_cast_safe = True
    torch.Tensor.to = _complex_cast_safe_to


_install_complex_cast_to_shim()
