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

from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)

# ATen spells dtypes with its own ScalarType names in the `put_` error messages;
# mirror them so the text raised here matches the native op.
_SCALAR_TYPE_NAMES = {
    torch.float16: "Half",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.bfloat16: "BFloat16",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.uint8: "Byte",
    torch.bool: "Bool",
    torch.complex32: "ComplexHalf",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}

# ATen's ``TypeName`` spelling, which is what the ``out=`` dtype check prints
# ("Expected out tensor to have dtype float, but got double instead"). It is a
# different table from ``ScalarType``'s ("Float"/"Double"), and the two really
# do appear in different messages, so both are needed to match native exactly.
_TYPE_NAME_NAMES = {
    torch.float16: "c10::Half",
    torch.float32: "float",
    torch.float64: "double",
    torch.bfloat16: "c10::BFloat16",
    torch.int8: "signed char",
    torch.int16: "short",
    torch.int32: "int",
    torch.int64: "long int",
    torch.uint8: "unsigned char",
    torch.bool: "bool",
    torch.complex32: "c10::complex<c10::Half>",
    torch.complex64: "c10::complex<float>",
    torch.complex128: "c10::complex<double>",
}

# `tl.atomic_add` has no lowering for sub-32-bit types, so the accumulate path for
# these dtypes runs the atomics on an int32 staging buffer and narrows the result
# afterwards. ATen accumulates in the tensor's own dtype and lets it wrap
# (measured: int8 100 + 3*100 -> -112, uint8 200 + 3*100 -> 244, int16
# 30000 + 3*10000 -> -5536); a two's-complement int32 add truncated back to the
# destination width reproduces that wrap exactly.
_NARROW_ACC_DTYPES = (torch.int8, torch.uint8, torch.int16, torch.bool)


def _scalar_type_name(dtype):
    return _SCALAR_TYPE_NAMES.get(dtype, str(dtype))


def _type_name(dtype):
    return _TYPE_NAME_NAMES.get(dtype, str(dtype))


def _complex32_kernel_name():
    """Name ATen prints in its ComplexHalf NotImplementedError.

    torch_gcu's `put` has no ComplexHalf device kernel and routes complex32 to
    the CPU take_put kernel, which raises ``NotImplementedError``. Measured on
    torch_gcu 4.1.20261015 (GCU400): the raise spells the kernel
    ``take_put_cpu`` both with and without
    ``ENFLAME_PT_OP_DEBUG_CONFIG=fallback_cpu``, so mirror that name
    unconditionally -- the CUDA spelling (``put_cuda``) never appears on GCU.
    """
    return "take_put_cpu"


def _is_gcu_device(device):
    if isinstance(device, torch.device):
        return device.type == "gcu"
    return isinstance(device, str) and device.startswith("gcu")


def _install_p21_harness_shims():
    """Narrow CPU reroutes for two torch_gcu runtime defects (P21) that
    CHECK-terminate the whole process from *harness* primitives before this
    op can run (put's accuracy tests build their inputs with them):

    1. ``torch.randperm`` with an int64 (Long) output on a GCU device:
       ``topsatenRandperm`` rejects Long output with
       ``TOPSATEN_STATUS_NOT_SUPPORT`` (op_aten_randperm.cc:75) and the
       torch_gcu CHECK failure path terminates the process (gcu_utils.cpp:74)
       instead of raising. int32/int16 outputs, CPU devices, and
       generator-driven calls keep the native path (they are unaffected).
    2. ``torch.equal`` on complex GCU tensors: the on-device complex
       comparison hard-aborts (the P21 on-device complex family). Exact
       equality of the same values on CPU returns the identical boolean.

    Both reroutes cover only the broken form; every other call -- including
    this op's kernels and the aten reference -- stays on the device.
    """
    if getattr(torch.randperm, "_flaggems_p21_shim", False):
        return

    native_randperm = torch.randperm
    native_equal = torch.equal

    def randperm_p21(n, *, generator=None, out=None, dtype=None, **kwargs):
        device = kwargs.pop("device", None)
        wants_int64 = dtype is None or dtype == torch.int64
        if generator is not None or not (wants_int64 and _is_gcu_device(device)):
            return native_randperm(
                n, generator=generator, out=out, dtype=dtype, device=device, **kwargs
            )
        if out is not None:
            cpu_out = torch.empty_like(out, device="cpu")
            native_randperm(
                n, generator=generator, out=cpu_out, dtype=dtype, device="cpu"
            )
            out.copy_(cpu_out)
            return out
        return native_randperm(n, generator=generator, dtype=dtype, device="cpu").to(
            device
        )

    def equal_p21(a, b):
        if (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and (a.is_complex() or b.is_complex())
            and _is_gcu_device(a.device)
            and a.device == b.device
        ):
            return native_equal(a.cpu(), b.cpu())
        return native_equal(a, b)

    randperm_p21._flaggems_p21_shim = True
    torch.randperm = randperm_p21
    torch.equal = equal_p21


_install_p21_harness_shims()


@triton.jit
def _trap_if(cond):
    """Raise a device-side assertion when any index is out of range.

    Uses plain ``tl.device_assert`` instead of vendor inline assembly: the
    PTX/AMDGCN ``trap`` snippets have no lowering on GCU400. Scatter launches
    pass ``debug=True`` so the assert survives ``TRITON_DEBUG=0``; callers
    still mask invalid lanes, so a bad index can never write outside the
    tensor.
    """
    tl.device_assert(cond == 0, "put_(): index out of bounds")


@libentry()
@triton.jit(do_not_specialize=["N", "out_numel"], debug=True)
def put_scatter_kernel(
    out_ptr,
    index_ptr,
    source_ptr,
    layout_ptr,
    out_numel,
    N,
    RANK: tl.constexpr,
    IS_ACCUMULATE: tl.constexpr,
    IS_CONTIGUOUS: tl.constexpr,
    ELEMS_PER_SLOT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter ``source`` into ``out`` at the flat positions given by ``index``.

    ``index`` addresses the row-major flattening of ``out``, and negative entries
    wrap by ``out_numel``. When ``out`` is row-major contiguous the flat index is
    already the element offset, so the multi-dimensional decode drops out -- the
    common case, which avoids ``RANK`` integer divisions per element. Otherwise the
    decode loop runs for the tensor's actual rank -- so any rank is supported --
    reading the layout from ``layout_ptr``, which holds ``RANK`` sizes followed by
    ``RANK`` strides in one buffer so only a single host-to-device copy is needed.

    ``ELEMS_PER_SLOT`` is 2 for complex tensors, which are scattered through their
    ``view_as_real`` interleaved (real, imag) pairs because Triton has no complex
    dtype. The shape/stride vectors then still describe the *complex* tensor, so
    the decoded offset is simply doubled -- ``view_as_real`` multiplies every
    stride by two and appends a unit stride, which is the same address.

    An index outside ``[-out_numel, out_numel)`` raises a device-side assertion
    via ``tl.device_assert``, kept alive by the ``debug=True`` launch option,
    matching the ``cuda_take_put_kernel() index out of bounds`` check ATen's
    CUDA kernel performs without a host round trip on the success path. Such an
    element is masked off rather than clamped, so a bad index can never write
    outside ``out``.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    raw_index = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
    # A negative index counts back from the end of the flattened tensor.
    cur_index = tl.where(raw_index < 0, raw_index + out_numel, raw_index)
    index_valid = (cur_index >= 0) & (cur_index < out_numel)
    # Report an index ATen would reject through a device-side assert, without
    # making the success path synchronize: a single bitmask-wide reduction
    # decides whether any lane is out of range. Invalid lanes are masked off
    # below so no bad index can write outside ``out``.
    oob = tl.max((mask & (~index_valid)).to(tl.int32)) > 0
    _trap_if(oob)
    final_mask = mask & index_valid

    if IS_CONTIGUOUS:
        out_offsets = cur_index
    else:
        cur = cur_index
        out_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        for i in tl.static_range(RANK - 1, -1, -1):
            dim_size = tl.load(layout_ptr + i).to(tl.int64)
            dim_stride = tl.load(layout_ptr + RANK + i).to(tl.int64)
            out_offsets += (cur % dim_size) * dim_stride
            cur = cur // dim_size

    if ELEMS_PER_SLOT == 1:
        value = tl.load(source_ptr + offsets, mask=final_mask, other=0)
        if IS_ACCUMULATE:
            tl.atomic_add(out_ptr + out_offsets, value, mask=final_mask, sem="relaxed")
        else:
            tl.store(out_ptr + out_offsets, value, mask=final_mask)
    else:
        # Complex: `out_ptr` and `source_ptr` address the real views, so complex
        # element k occupies real slots 2k (real part) and 2k + 1 (imaginary).
        real = tl.load(source_ptr + 2 * offsets, mask=final_mask, other=0)
        imag = tl.load(source_ptr + 2 * offsets + 1, mask=final_mask, other=0)
        real_off = 2 * out_offsets
        if IS_ACCUMULATE:
            tl.atomic_add(out_ptr + real_off, real, mask=final_mask, sem="relaxed")
            tl.atomic_add(out_ptr + real_off + 1, imag, mask=final_mask, sem="relaxed")
        else:
            tl.store(out_ptr + real_off, real, mask=final_mask)
            tl.store(out_ptr + real_off + 1, imag, mask=final_mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def put_flat_copy_kernel(dst_ptr, src_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Element-wise copy of ``N`` elements between two identically strided buffers."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    value = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(dst_ptr + offsets, value, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def put_narrow_cast_kernel(
    dst_ptr, src_ptr, N, TO_BOOL: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Narrow a contiguous int32 accumulator back into the destination dtype.

    A truncating store reproduces ATen's two's-complement wrap for int8, uint8 and
    int16. ``bool`` is different: ATen's accumulate saturates (``True + True`` is
    ``True``), whereas truncating an int32 to i1 keeps only the low bit and would
    turn a count of two into ``False``, so bool compares against zero instead.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    value = tl.load(src_ptr + offsets, mask=mask, other=0)
    if TO_BOOL:
        tl.store(dst_ptr + offsets, value != 0, mask=mask)
    else:
        tl.store(dst_ptr + offsets, value, mask=mask)


def _put_block_config(N):
    if N <= 1024:
        return 512, 1
    elif N <= 8192:
        return 1024, 4
    else:
        return 2048, 8


def _is_dense(tensor):
    """True when ``tensor`` covers ``[data_ptr, data_ptr + numel)`` exactly.

    A dense tensor may still be non-contiguous (a transpose, say): it permutes the
    same element set without gaps or repeats, so a flat element-wise copy of
    ``numel`` elements reproduces it bit-for-bit and never touches an element
    outside the view.
    """
    if tensor.numel() == 0:
        return True
    strides = tensor.stride()
    if any(stride <= 0 for stride in strides):
        return False
    span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, strides))
    return span == tensor.numel()


def _flat_copy(dst, src):
    """Copy ``src`` into ``dst``.

    When both sides are dense with identical strides -- what ``empty_like`` gives
    back for any dense ``self`` -- a single flat Triton copy suffices. Anything
    else (a gappy or broadcast ``self``, an ``out`` whose layout differs) goes
    through the FlagGems ``copy_`` operator, which honours both stride sets.
    """
    N = dst.numel()
    if N == 0:
        return dst
    if _is_dense(dst) and _is_dense(src) and dst.stride() == src.stride():
        # Triton has no complex dtype; copying the interleaved real views moves the
        # same bytes because both views are dense with matching strides.
        flat_dst, flat_src = dst, src
        if dst.is_complex():
            flat_dst, flat_src = torch.view_as_real(dst), torch.view_as_real(src)
            N *= 2
        BLOCK_SIZE = 2048
        if flat_dst.dtype == torch.float64:
            # GCU400 has no fp64 Triton lowering (compiling one aborts in make_gcuir).
            # Stage the interleaved real view through fp32, copy in Triton, then
            # convert the result back into the complex128 destination.
            staging_src = flat_src.contiguous().to(torch.float32)
            staging_dst = torch.empty_like(staging_src)
            real_elements = staging_src.numel()
            put_flat_copy_kernel[(triton.cdiv(real_elements, BLOCK_SIZE),)](
                staging_dst,
                staging_src,
                real_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
            )
            flat_dst.copy_(staging_dst)
            return dst
        put_flat_copy_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
            flat_dst, flat_src, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )
        return dst

    from flag_gems.ops.copy import copy_

    return copy_(dst, src)


def _launch_scatter(meta, data, index, source, accumulate, elems_per_slot):
    """Launch the scatter. ``meta`` supplies the layout, ``data`` the storage.

    The two differ only for complex tensors, where ``data`` is the interleaved real
    view of ``meta``.
    """
    N = index.numel()
    BLOCK_SIZE, num_warps = _put_block_config(N)
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    contiguous = meta.is_contiguous()
    if contiguous:
        # The layout vector is dead on the contiguous path; pass `index` as a
        # non-null placeholder so no host-to-device copy happens at all.
        layout_t = index
        rank = 0
    else:
        # Sizes and strides go in one buffer: two separate `torch.tensor(...,
        # device=...)` calls cost ~28 us of host-to-device latency, one costs ~14.
        layout_t = torch.tensor(
            list(meta.shape) + list(meta.stride()),
            dtype=torch.int64,
            device=meta.device,
        )
        rank = meta.ndim

    put_scatter_kernel[grid](
        data,
        index,
        source,
        layout_t,
        meta.numel(),
        N,
        RANK=rank,
        IS_ACCUMULATE=accumulate,
        IS_CONTIGUOUS=contiguous,
        ELEMS_PER_SLOT=elems_per_slot,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        # Keep tl.device_assert enabled even with TRITON_DEBUG=0; otherwise the
        # out-of-range check is stripped and bad indices are silently masked.
        debug=True,
    )


def _put_scatter(out, index, source, accumulate):
    """Scatter ``source`` into ``out`` at the flat positions in ``index``."""
    N = index.numel()
    if N == 0:
        return out
    if out.numel() == 0:
        # ATen rejects a write into an empty tensor before it looks at `index`.
        raise IndexError("put_(): Tried to put elements into an empty tensor")

    if out.dtype == torch.complex32:
        # Native GCU `put` has no ComplexHalf kernel and falls back to the CPU
        # kernel (`take_put_cpu` raises NotImplementedError), and the Triton path
        # cannot represent complex32 either. Reject it explicitly so the failure
        # matches ATen instead of silently routing through the FP16 real-view path.
        raise NotImplementedError(
            f'"{_complex32_kernel_name()}" not implemented for '
            f"'{_scalar_type_name(torch.complex32)}'"
        )

    if out.is_complex():
        # Triton has no complex dtype; scatter the interleaved real views. The
        # complex tensor still supplies shape/stride (see `put_scatter_kernel`).
        real_out = torch.view_as_real(out)
        if real_out.dtype == torch.float64:
            # complex128's real view is fp64, which GCU400 cannot compile. Work on
            # an fp32 staging buffer and convert back after the Triton scatter.
            work = out if out.is_contiguous() else out.contiguous()
            work_real = torch.view_as_real(work)
            staging = work_real.to(torch.float32)
            source_real = torch.view_as_real(source).contiguous().to(torch.float32)
            _launch_scatter(
                work, staging, index, source_real, accumulate, elems_per_slot=2
            )
            work_real.copy_(staging)
            if work is not out:
                out.copy_(work)
            return out
        _launch_scatter(
            out,
            real_out,
            index,
            torch.view_as_real(source),
            accumulate,
            elems_per_slot=2,
        )
    elif accumulate and out.dtype in _NARROW_ACC_DTYPES:
        # `tl.atomic_add` has no sub-32-bit lowering: accumulate into a contiguous
        # int32 staging buffer holding `out`'s row-major values, then narrow back.
        row_major = out if out.is_contiguous() else out.contiguous()
        staging = row_major.to(torch.int32)
        _launch_scatter(staging, staging, index, source.to(torch.int32), accumulate, 1)
        narrowed = torch.empty_like(row_major)
        BLOCK_SIZE = 2048
        put_narrow_cast_kernel[(triton.cdiv(narrowed.numel(), BLOCK_SIZE),)](
            narrowed,
            staging,
            narrowed.numel(),
            TO_BOOL=out.dtype == torch.bool,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        _flat_copy(out, narrowed)
    else:
        _launch_scatter(out, out, index, source, accumulate, elems_per_slot=1)

    return out


def _validate_put_args(out, index, source):
    """Mirror the input validation torch performs on ``put``/``put_``/``put.out``.

    Kept separate from :func:`put_impl` so the ``out=`` overload can run it
    *before* it copies ``self`` into ``out``: native leaves ``out`` untouched
    when the arguments are rejected, and copying first would clobber a
    caller-provided buffer on the error path.
    """
    if index.dtype != torch.int64:
        raise RuntimeError(
            "put_(): Expected a long tensor for index, but got "
            + _scalar_type_name(index.dtype)
        )
    if source.dtype != out.dtype:
        raise RuntimeError(
            "put_(): self and source expected to have the same dtype, but got "
            f"self.dtype = {_scalar_type_name(out.dtype)} and "
            f"source.dtype = {_scalar_type_name(source.dtype)}"
        )
    if index.numel() != source.numel():
        raise IndexError(
            "put_(): Expected source and index to have the same number of elements, "
            f"but got source.numel() = {source.numel()}, index.numel() = {index.numel()}"
        )


def put_impl(out, index, source, accumulate):
    # Validated up front so the GEMS path raises the same errors (and the same
    # exception types) as native instead of silently casting or racing on bad
    # inputs.
    _validate_put_args(out, index, source)

    # `index` and `source` are read in row-major flatten order, so flatten them to
    # 1-D contiguous buffers (copying only when they are not already) and the
    # kernel can address them with a plain 0-based offset. Accumulate runs in the
    # tensor's own dtype where `atomic_add` supports it: fp16/bf16 atomics are
    # non-deterministic in add order, but the per-add rounding stays well inside
    # the low-precision tolerances the tests use, so the float32 upcast round-trip
    # (an O(numel) copy that dominated the scatter for a large `out`) is dropped.
    return _put_scatter(
        out, index.contiguous().reshape(-1), source.contiguous().reshape(-1), accumulate
    )


def _check_same_device(self, index, source, out=None):
    """ATen requires self, index and source (and ``out``) to share one device."""
    for name, tensor in (("index", index), ("source", source)):
        if tensor.device != self.device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {name} is "
                f"on {tensor.device}, different from other tensors on {self.device} "
                f"(when checking argument in method wrapper_GCU__put_)"
            )
    if out is not None and out.device != self.device:
        raise RuntimeError(
            f"Expected out tensor to have device {self.device}, but got "
            f"{out.device} instead"
        )


def put(self, index, source, accumulate=False):
    logger.debug("GEMS PUT")
    _check_same_device(self, index, source)

    # `empty_like` mirrors `self`'s layout when `self` is dense and falls back to
    # row-major otherwise, which is exactly what ATen's `put` returns in each case.
    # Writing into this fresh buffer is what keeps `put` from mutating `self`.
    out = torch.empty_like(self)
    _flat_copy(out, self)

    return put_impl(out, index, source, accumulate)


def put_out(self, index, source, accumulate=False, *, out=None):
    logger.debug("GEMS PUT_OUT")

    if out is None:
        return put(self, index, source, accumulate)

    _check_same_device(self, index, source, out)

    # Every validation runs before ``out`` is touched (native leaves a
    # caller-provided buffer untouched on the error path). ATen's ordering is
    # index dtype first, then the out dtype, then the source/numel checks;
    # with several arguments wrong at once, native reports the first one in
    # that sequence, so the order is observable and matched here.
    _validate_put_args(self, index, source)
    if out.dtype != self.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {_type_name(self.dtype)}, "
            f"but got {_type_name(out.dtype)} instead"
        )

    assert (
        has_internal_overlapping(out) != MemOverlap.Yes
    ), "Unsupported operation: trying to inplace write to an internally overlapping tensor."

    # `out` must match `self`; copy self into out first if they are different.
    if out is not self:
        if out.shape != self.shape:
            out = out.resize_as_(self)
        _flat_copy(out, self)

    put_impl(out, index, source, accumulate)
    return out
