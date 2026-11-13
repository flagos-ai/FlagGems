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
import sys

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

# GCU hardware grid limits for the linearized tile split in
# ``quantized_max_pool2d_kernel`` (grid.x <= 65535, grid.y <= 255).
_MAX_GRID_X = 65535

# The quantized integer dtypes ``aten::quantized_max_pool2d`` dispatches over
# (its kernel body is wrapped in ``AT_DISPATCH_QINT_TYPES_AND(Byte, ...)``),
# mapped to the dtype of their ``int_repr()``. Pooling runs directly on the
# integer representation, so the kernel only needs their width/signedness.
_QINT_DTYPES = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}

# ATen reports the *C++* spelling of a dtype in the ``out=`` dtype error, e.g.
# "Expected out tensor to have dtype c10::quint8, but got c10::qint8 instead".
# Reproduce the spellings so the message matches the native operator verbatim.
_ATEN_DTYPE_NAMES = {
    torch.quint8: "c10::quint8",
    torch.qint8: "c10::qint8",
    torch.qint32: "c10::qint32",
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "c10::Half",
    torch.bfloat16: "c10::BFloat16",
    torch.bool: "bool",
    torch.uint8: "unsigned char",
    torch.int8: "signed char",
    torch.int16: "short int",
    torch.int32: "int",
    torch.int64: "long int",
    torch.complex64: "c10::complex<float>",
    torch.complex128: "c10::complex<double>",
}


def _aten_dtype_name(dtype: torch.dtype) -> str:
    return _ATEN_DTYPE_NAMES.get(dtype, str(dtype))


def max_pool2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    """Compute one spatial dimension of the 2-D max-pool output.

    Mirrors ``at::native::pooling_output_shape``, including the two checks it
    performs before computing the size and the ``ceil_mode`` correction that
    keeps the last pooling window starting inside the image.
    """
    if stride == 0:
        raise RuntimeError("stride should not be zero")
    if padding < 0:
        raise RuntimeError(f"pad must be non-negative, but got pad: {padding}")

    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        # Python's floor division matches ATen's ``div_rtn`` for negatives too.
        output_size = (numerator + stride - 1) // stride + 1
        # PyTorch-compatible adjustment for ceil_mode: ensure that the last
        # pooling window starts inside the image.
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1

    return output_size


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("quantized_max_pool2d"),
    key=["out_h", "out_w", "kernel_h", "kernel_w", "stride_h", "stride_w"],
)
@triton.jit
def quantized_max_pool2d_kernel(
    input_ptr,
    output_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Output tensor strides
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    # Input/Output shapes
    in_n,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Pooling parameters
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Meta-parameters for tiling
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Forward kernel for quantized 2-D max pooling.

    The kernel operates directly on the integer representation (``int_repr``) of
    a per-tensor quantized tensor -- ``uint8`` for ``quint8``, ``int8`` for
    ``qint8``, ``int32`` for ``qint32``. Because the quantization mapping
    ``real = (q - zero_point) * scale`` is monotonically increasing in ``q`` for
    a positive ``scale``, taking the max over the quantized integers is
    equivalent to taking the max over the real values. The identity of the max
    is what is stored, so no requantization is needed and the result is exact
    for every one of the three dtypes.

    Positions that fall outside the image are filled with the minimum of the
    *underlying integer* dtype, which is the same sentinel ATen's
    ``spatial_dilated_max_pooling`` uses
    (``std::numeric_limits<T::underlying>::lowest()``).

    Both sides are addressed through explicit strides, so a non-contiguous
    input needs no repacking and a channels-last output can be written in place.

    Grid: the (N * C, cdiv(out_h, BLOCK_H) * cdiv(out_w, BLOCK_W)) tile space is
    linearized (``pid_nc + pid_hw * (N * C)``, the original column-major order)
    and split across a 2-D launch so neither grid dimension exceeds the GCU
    limits (grid.x <= 65535, grid.y <= 255).  A batch like (128, 512, 7, 7)
    alone puts N * C = 65536 past the grid.x limit; the linearized split covers
    it (and any tile count up to 65535 * 255) with one program per tile and the
    same tile order as the original grid.
    """
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    num_spatial = num_h_blocks * num_w_blocks
    num_nc = in_n * in_c
    total_blocks = num_nc * num_spatial
    # Must match the host grid(): min(65535, total_blocks)
    grid_x = tl.minimum(65535, total_blocks)
    linear_pid = tl.program_id(0) + tl.program_id(1) * grid_x
    if linear_pid >= total_blocks:
        return
    # Same tile order as the original (N * C, spatial) grid.
    pid_nc = linear_pid % num_nc
    pid_hw = linear_pid // num_nc
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
            input_offset = h_in * in_stride_h + w_in * in_stride_w
            current_val = tl.load(
                input_base_ptr + input_offset, mask=in_mask, other=min_val
            )
            max_val_acc = tl.maximum(max_val_acc, current_val)

    out_base_ptr = output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    output_block_ptr = (
        out_base_ptr
        + h_out_offsets[:, None] * out_stride_h
        + w_out_offsets[None, :] * out_stride_w
    )

    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)


def _check_maxpool2d_params(kernel_size, stride, padding, dilation):
    """Mirror ATen's ``check_maxpool2d_params`` argument-shape validation.

    Returns the parameters expanded to 2-tuples. ``stride`` may be an empty
    sequence, in which case ``None`` is returned for it so the caller can
    substitute ``kernel_size`` (what ATen does right after this check).

    ATen accepts 1- *or* 2-element lists for kernel_size/padding/dilation, but
    then indexes ``[0]`` and ``[1]`` unconditionally, so a 1-element list reads
    past the end of the array. We reject that instead of reproducing the
    out-of-bounds read; a bare Python int reaches the ``int[2]`` schema slot
    broadcast to both dims, which is the well-defined way to say "same value".
    """

    def _length_checked(param, name, message):
        if isinstance(param, int):
            # A bare int reaches the ATen schema as ``int[2]``, i.e. broadcast.
            return param, param
        if isinstance(param, (list, tuple)):
            if len(param) != 2:
                raise RuntimeError(f"{message}{len(param)}")
            return tuple(int(p) for p in param)
        raise RuntimeError(f"Invalid {name}: {param}")

    kernel = _length_checked(
        kernel_size, "kernel_size", "Expected 1d or 2d kernel size, got "
    )

    if isinstance(stride, (list, tuple)) and len(stride) == 0:
        strides = None
    elif stride is None:
        # The schema default is ``int[2] stride=[]``; ``None`` from a Python
        # caller means the same thing as an omitted argument.
        strides = None
    else:
        # Note the missing space after "got": ATen's message really does read
        # "Expected no strides or 2d strides, got3".
        strides = _length_checked(
            stride, "stride", "Expected no strides or 2d strides, got"
        )

    pads = _length_checked(padding, "padding", "Expected 1d or 2d padding, got ")
    dilations = _length_checked(
        dilation, "dilation", "Expected 1d or 2d dilation, got "
    )

    if any(d < 1 for d in dilations):
        raise RuntimeError("Expected dilation >= 1")

    return kernel, strides, pads, dilations


def _parse_pool_params(kernel_size, stride, padding, dilation):
    """Parse and validate 2-D pooling parameters the way ATen does.

    Each parameter can be an int (applied to both spatial dims) or a 2-element
    tuple/list (H, W). An empty or omitted ``stride`` falls back to
    ``kernel_size``, matching the ``int[2] stride=[]`` schema default of
    ``aten::quantized_max_pool2d``.
    """
    kernel, strides, pads, dilations = _check_maxpool2d_params(
        kernel_size, stride, padding, dilation
    )
    if strides is None:
        strides = kernel

    kernel_h, kernel_w = kernel
    stride_h, stride_w = strides
    padding_h, padding_w = pads
    dilation_h, dilation_w = dilations

    # ATen validates kernel/stride positivity inside ``q_maxpool_2d``, in this
    # order, before it looks at the input tensor at all.
    if kernel_h <= 0 or kernel_w <= 0:
        raise RuntimeError("kernel_size should be greater than zero.")
    if stride_h <= 0 or stride_w <= 0:
        raise RuntimeError("strides should be greater than zero.")

    return (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )


# Backwards-compatible alias: the benchmark and earlier revisions of this file
# import ``_parse_pool_params`` under this name.
_parse_param_pairs = _parse_pool_params


def _check_quantized_input(input):
    """Reject inputs ATen's ``quantized_max_pool2d`` cannot accept.

    ATen runs this *after* the parameter and shape checks: it only touches the
    quantizer when it reaches ``qx.q_scale()``, so a per-channel input with a
    bad kernel size reports the kernel size, not the qscheme. Keep the same
    order by calling this last.
    """
    if not input.is_quantized:
        raise RuntimeError(
            "quantized_max_pool2d expects a quantized tensor "
            "(torch.quint8, torch.qint8 or torch.qint32)"
        )
    if input.qscheme() != torch.per_tensor_affine:
        # ATen's TORCH_CHECK inside the quantizer, verbatim (the trailing
        # boilerplate about improving the message is appended by TORCH_CHECK
        # itself and is not reproduced here).
        raise RuntimeError(
            "Expected quantizer->qscheme() == kPerTensorAffine to be true, "
            "but got false."
        )
    if input.dtype not in _QINT_DTYPES:
        raise RuntimeError(
            f"quantized_max_pool2d not implemented for '{input.dtype}'; "
            "expected one of torch.quint8, torch.qint8, torch.qint32"
        )


def _quantized_max_pool2d_output_shape(input, params, ceil_mode):
    """Validate the input against ``params`` and return the output shape.

    Reproduces the remaining checks of ATen's ``q_maxpool_2d``, in order: rank
    (3 or 4), non-zero C/H/W, padding vs. half the kernel, and finally a
    positive output size. A 3-D input is the native (C, H, W) form -- ATen
    treats it as a single batch.
    """
    (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    ) = params

    ndim = input.dim()
    if ndim not in (3, 4):
        raise RuntimeError("Expecting the input tensor of rank 3 or 4.")

    in_n = input.shape[0] if ndim == 4 else 1
    in_c, in_h, in_w = input.shape[-3:]
    # ATen only requires C/H/W to be non-zero; an empty batch is legal.
    if in_c <= 0 or in_h <= 0 or in_w <= 0:
        raise RuntimeError("input dimensions must be non-zero.")
    if kernel_h // 2 < padding_h or kernel_w // 2 < padding_w:
        raise RuntimeError("padding should be smaller than half of kernel_size.")

    out_h = max_pool2d_output_size(
        in_h, kernel_h, stride_h, padding_h, dilation_h, ceil_mode
    )
    out_w = max_pool2d_output_size(
        in_w, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
    )
    if out_h <= 0 or out_w <= 0:
        raise RuntimeError(
            f"Given input size: ({in_c}x{in_h}x{in_w}). "
            f"Calculated output size: ({in_c}x{out_h}x{out_w}). "
            "Output size is too small."
        )

    if ndim == 3:
        return (in_c, out_h, out_w)
    return (in_n, in_c, out_h, out_w)


def _int_view(qtensor):
    """Alias a quantized tensor's storage as a writable integer tensor.

    ``int_repr()`` returns a *copy*, so it cannot be used as a kernel
    destination, and for a strided input it would materialise a compacted copy.
    Aliasing the storage with the tensor's own sizes and strides gives a view
    the kernel can read from or store into, whatever layout it has.
    """
    return torch.empty(
        0, dtype=_QINT_DTYPES[qtensor.dtype], device=qtensor.device
    ).set_(
        qtensor.untyped_storage(),
        qtensor.storage_offset(),
        tuple(qtensor.shape),
        qtensor.stride(),
    )


def _as_4d(int_tensor):
    """View a 3-D (C, H, W) integer tensor as (1, C, H, W)."""
    if int_tensor.dim() == 3:
        return int_tensor.unsqueeze(0)
    return int_tensor


def _launch_quantized_max_pool2d(input_int, out_int, params):
    """Run the pooling kernel over an integer view of the input."""
    (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    ) = params

    input_int = _as_4d(input_int)
    out_int = _as_4d(out_int)
    in_n, in_c, in_h, in_w = input_int.shape
    _, _, out_h, out_w = out_int.shape

    def grid(meta):
        num_h_blocks = triton.cdiv(out_h, meta["BLOCK_H"])
        num_w_blocks = triton.cdiv(out_w, meta["BLOCK_W"])
        total = in_n * in_c * num_h_blocks * num_w_blocks
        # Linearized tile split: keep grid.x within the 65535 hardware limit
        # (N * C alone reaches 65536 for (128, 512, 7, 7)) and grid.y within
        # 255; the kernel walks the same column-major tile order.  Must match
        # the kernel's own ``grid_x = min(65535, total_blocks)``.
        grid_x = min(_MAX_GRID_X, total)
        grid_y = triton.cdiv(total, grid_x)
        return (grid_x, grid_y)

    quantized_max_pool2d_kernel[grid](
        input_int,
        out_int,
        *input_int.stride(),
        *out_int.stride(),
        in_n,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )
    return out_int


def _output_memory_format(input):
    """Pick the memory format the native kernel for this device allocates with.

    The kernel this operator replaces is ``quantized_max_pool2d_cudnn``, not the
    CPU one, and the two disagree. cuDNN has no strided or 3-D convolution path,
    so its wrapper in ``QuantizedMaxPool2d.cpp`` permutes *every* 4-D input to
    channels-last first -- NCHW, strided and already-channels-last alike -- and
    always allocates the result with ``memory_format=ChannelsLast``. A 3-D
    (C, H, W) input is instead viewed as a single batch and stays
    NCHW-contiguous.

    The CPU kernel only produces a channels-last result when the input already
    has that layout, and gives an NCHW-contiguous result otherwise (including
    for a degenerate ``C == 1`` shape, where a channels-last allocation would
    have different strides). That rule is kept for a CPU input, so a direct call
    with CPU tensors stays layout-identical to the CPU operator.

    The rule keys on the device rather than the dtype: ``quantized_max_pool2d``
    is registered for the QuantizedCUDA dispatch key, where a 4-D input always
    reaches the cuDNN path, and cuDNN only handles ``qint8``. For
    ``quint8``/``qint32`` the native CUDA op raises
    ("TensorDescriptor does not support ..."), so there is no native layout to
    contradict, and giving every 4-D accelerator input the same channels-last
    layout avoids a result whose strides depend on the dtype.
    """
    if input.dim() != 4:
        return torch.contiguous_format
    if input.device.type != "cpu":
        return torch.channels_last
    if (
        input.is_contiguous(memory_format=torch.channels_last)
        and not input.is_contiguous()
    ):
        return torch.channels_last
    return torch.contiguous_format


def quantized_max_pool2d(
    input: torch.Tensor,
    kernel_size,
    stride=(),
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Apply 2-D max pooling to a per-tensor quantized tensor.

    Accepts both native input forms -- 4-D (N, C, H, W) and 3-D (C, H, W) --
    and the three dtypes the native operator dispatches over:
    ``torch.quint8``, ``torch.qint8`` and ``torch.qint32``. Pooling is
    performed directly over the quantized integer representation, which (for a
    positive scale) yields the same result as pooling over the dequantized
    values while keeping the input's ``scale``/``zero_point`` on the output.

    An omitted or empty ``stride`` falls back to ``kernel_size``, matching the
    ``int[2] stride=[]`` schema default of ``aten::quantized_max_pool2d``.
    """
    logger.debug("GEMS QUANTIZED_MAX_POOL2D")

    params = _parse_pool_params(kernel_size, stride, padding, dilation)
    out_shape = _quantized_max_pool2d_output_shape(input, params, ceil_mode)
    _check_quantized_input(input)

    scale = float(input.q_scale())
    zero_point = int(input.q_zero_point())

    # Allocate the quantized result directly: ``_empty_affine_quantized``
    # honours ``memory_format``, whereas wrapping a strided integer tensor with
    # ``_make_per_tensor_quantized_tensor`` silently normalises its strides on
    # CUDA and would drop the channels-last layout.
    out = torch._empty_affine_quantized(
        out_shape,
        scale=scale,
        zero_point=zero_point,
        dtype=input.dtype,
        device=input.device,
        memory_format=_output_memory_format(input),
    )

    if out.numel() != 0:
        _launch_quantized_max_pool2d(_int_view(input), _int_view(out), params)

    return out


def _contiguous_strides(shape):
    strides = []
    acc = 1
    for size in reversed(shape):
        strides.append(acc)
        acc *= size
    return tuple(reversed(strides)), acc


def _resize_quantized_out(out, shape):
    """Resize a quantized ``out`` tensor to ``shape``, ATen-style.

    ``aten::resize_`` and ``aten::set_.source_Tensor`` are both unimplemented
    for the QuantizedCUDA backend, so the metadata is rewritten through
    ``set_(storage, offset, size, stride)``, which is registered. This
    reproduces what ``at::native::resize_output`` does for a strided tensor:
    the storage offset is preserved, the storage only ever grows, and the
    resized tensor is given contiguous strides.
    """
    strides, numel = _contiguous_strides(tuple(shape))

    storage = out.untyped_storage()
    offset = out.storage_offset()
    needed = (offset + numel) * out.element_size()
    if numel != 0 and storage.size() < needed:
        storage.resize_(needed)
    out.set_(storage, offset, tuple(shape), strides)
    return out


def _aliases(input, out):
    """True when ``out`` may share storage with ``input``."""
    return out.untyped_storage().data_ptr() == input.untyped_storage().data_ptr()


def _same_quantizer(input, out):
    """True when ``out`` already carries ``input``'s per-tensor quantizer.

    A kernel write moves integers only, so it cannot re-point ``out``'s
    quantizer the way ``copy_`` does; pooling in place is only equivalent when
    ``out`` is per-tensor affine with the same ``scale`` and ``zero_point``.
    """
    return (
        out.qscheme() == torch.per_tensor_affine
        and out.q_scale() == input.q_scale()
        and out.q_zero_point() == input.q_zero_point()
    )


def _adopt_quantizer(out, source):
    """Re-point ``out`` at ``source``'s per-tensor quantizer, in place.

    ``out.copy_(source)`` is the only ATen operation that moves a quantizer
    between quantized tensors, but it moves the values too. Since the values
    are written by the kernel (or copied separately through ``_copy_int``),
    only the quantizer needs to move, and that can be done with ``.data``:
    assigning an alias that shares ``out``'s storage, offset, sizes and strides
    swaps the quantizer while leaving the tensor's storage, layout and identity
    untouched -- the caller's ``out`` object is mutated in place, and a strided
    or offset view keeps writing into its own slice of the buffer.

    The alias is built with ``_empty_affine_quantized`` + ``set_`` because
    ``_make_per_tensor_quantized_tensor`` (and ``Tensor.to``) silently
    normalises the strides of a quantized tensor on CUDA, which would lose a
    channels-last layout.
    """
    alias = torch._empty_affine_quantized(
        0,
        scale=float(source.q_scale()),
        zero_point=int(source.q_zero_point()),
        dtype=out.dtype,
        device=out.device,
    )
    alias.set_(
        out.untyped_storage(),
        out.storage_offset(),
        tuple(out.shape),
        out.stride(),
    )
    out.data = alias
    return out


def _copy_int(dst_int, src_int):
    """Copy integers device-side, honouring both sides' strides.

    ``dst_int`` / ``src_int`` are the integer views produced by ``_int_view``,
    so the tensors are plain integer tensors and reach FlagGems' Triton
    ``copy_`` instead of being redispatched to ATen the way a quantized
    ``Tensor.copy_`` is.
    """
    from flag_gems.ops.copy import copy_

    copy_(dst_int, src_int)
    return dst_int


def quantized_max_pool2d_out(
    input: torch.Tensor,
    kernel_size,
    stride=(),
    padding=0,
    dilation=1,
    ceil_mode=False,
    *,
    out: torch.Tensor,
):
    """``out`` variant of :func:`quantized_max_pool2d`.

    Follows the contract of the autogenerated ``quantized_max_pool2d.out``
    kernel, which is written in terms of the functional one:

    .. code-block:: cpp

        auto tmp_output = at::quantized_max_pool2d(self, ...);
        resize_out_helper(out, tmp_output);   // resize_output(out, tmp.sizes())
        copy_arg(out, tmp_output);            // dtype check, device check, copy_

    So the shape, dtype and device contract matches ATen's: the result is
    computed *first* (which is also what makes an ``out`` that aliases the input
    safe), then ``out`` is resized to the result's shape, then its dtype and
    device are checked, and finally the values are written through ``out``'s own
    strides and the input's ``scale`` / ``zero_point`` are carried over. A
    mismatched-dtype ``out`` is therefore still resized before the error
    surfaces, exactly as in ATen.

    The final write differs from ATen in *how*, not in what lands where: the
    pooling kernel is strided on both sides, so when ``out`` does not overlap
    the input it stores the pooled integers through ``out`` directly, and only
    the quantizer is re-pointed (:func:`_adopt_quantizer`, no ``copy_`` at
    all). The aliasing case -- where ``out`` must keep the input's values until
    the result is complete -- copies the integer representations device-side
    through FlagGems' Triton ``copy_`` (:func:`_copy_int`). Neither path calls
    the native quantized ``copy_``: ``Tensor.copy_`` on quantized tensors
    redispatches to ATen, which would add a native O(n) pass after the kernel.
    """
    logger.debug("GEMS QUANTIZED_MAX_POOL2D_OUT")

    # The functional result's shape, dtype and device are all known without
    # running the kernel, so the resize and the two checks happen in ATen's
    # order either way; computing first is only *required* when ``out`` aliases
    # the input, where the resize would otherwise reshape the input out from
    # under the pooling.
    params = _parse_pool_params(kernel_size, stride, padding, dilation)
    out_shape = _quantized_max_pool2d_output_shape(input, params, ceil_mode)
    _check_quantized_input(input)

    result = None
    if _aliases(input, out):
        result = quantized_max_pool2d(
            input, kernel_size, stride, padding, dilation, ceil_mode
        )

    if tuple(out.shape) != tuple(out_shape):
        _resize_quantized_out(out, out_shape)

    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {_aten_dtype_name(input.dtype)}, "
            f"but got {_aten_dtype_name(out.dtype)} instead"
        )
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor to have device {input.device}, "
            f"but got {out.device} instead"
        )

    if out.qscheme() != torch.per_tensor_affine:
        # ATen's quantized ``copy_`` rejects a per-channel destination with this
        # message. It is raised by the copy, which runs after the resize and the
        # dtype/device checks, so it comes last here too.
        raise RuntimeError("Quantized Copy only works with same qscheme")

    if result is None:
        # ``out`` cannot overlap the input, so the pooled integers can be
        # written straight through ``out``'s own strides and storage offset --
        # the kernel is strided on both sides, so an ``out`` in any layout,
        # including a non-contiguous view, receives exactly the elements
        # ``out.copy_(functional_result)`` would have placed. Only the
        # quantizer still has to move, which :func:`_adopt_quantizer` does
        # without a copy.
        if out.numel() != 0:
            _launch_quantized_max_pool2d(_int_view(input), _int_view(out), params)
        if _same_quantizer(input, out):
            return out
        return _adopt_quantizer(out, input)

    # ``out`` aliases the input, so pooling cannot write into it directly: the
    # functional result was computed into ``result`` first. Moving those
    # integers into ``out`` is the one case that genuinely needs a copy; it is
    # done over the integer views, which lands in FlagGems' Triton ``copy_``.
    # A quantized ``out.copy_(result)`` would redispatch to ATen and run an
    # extra native O(n) pass after the Triton kernel had already run.
    if out.numel() != 0:
        _copy_int(_int_view(out), _int_view(result))
    return _adopt_quantizer(out, result)


# ---------------------------------------------------------------------------
# GCU dispatch bridge: per-tensor / per-channel quantized aten surface
# ---------------------------------------------------------------------------
# torch_gcu ships no kernels under the ``QuantizedPrivateUse1`` dispatch key
# -- the key every quantized GCU tensor dispatches through (the dispatcher
# renders it as the 'Quantizedgcu' backend) -- and PyTorch registers its
# quantized aten ops only for ``QuantizedCPU`` in this build.  Every quantized
# aten call the operator's accuracy harness needs therefore raises
# ``NotImplementedError`` before any FlagGems code runs: the input factory
# chain (``torch._make_per_tensor_quantized_tensor`` -> ``aten::empty_quantized``
# -> ``aten::_copy_from``), the direct ``torch._empty_affine_quantized``
# allocations, the metadata readers (``q_scale`` / ``q_zero_point`` /
# ``qscheme``), ``int_repr``, the quantized ``copy_`` worker and the
# dispatched ``torch.quantized_max_pool2d`` itself.  The generic FlagGems
# config registers the quantized factories and pooling ops under
# ``QuantizedCUDA``, which is the correct key only on NVIDIA devices and
# never matches a GCU tensor.
#
# The bridge below installs the minimal quantized aten surface on the keys
# GCU tensors actually dispatch to, at import time and fail-soft (a failed
# slot leaves the stock behaviour untouched), mirroring the established
# sparse accessor bridge in ``sparse_compressed_tensor.py``:
#
# * the metadata readers and ``set_`` redispatch to the ``QuantizedCPU``
#   kernels, which only touch TensorImpl / quantizer metadata and are
#   device independent;
# * the factories allocate through the ``QuantizedCPU`` kernels as well --
#   those honour the ``device`` argument, so the native allocator runs
#   exactly once on the requested (GCU) device;
# * value movement (``aten::_copy_from``) runs over integer views of the
#   quantized storage, landing on the plain-tensor copy kernels torch_gcu
#   already provides (including host <-> device);
# * ``int_repr`` / ``dequantize`` route to this backend's Triton
#   implementations, while ``clone`` / ``as_strided`` / ``set_`` and the
#   quantize ops are small host-side bridges (see their docstrings);
# * ``quantized_max_pool2d`` (and its ``.out`` overload) route to this
#   module, so a dispatched ``torch.quantized_max_pool2d`` on a GCU tensor
#   reaches the Triton pooling kernel.
#
# ``use_gems()`` stays additive: it registers under other keys
# (``PrivateUse1`` / ``QuantizedCUDA``), and a slot it cannot take is skipped
# with the registrar's standard warning.

_QUANTIZED_CPU_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.QuantizedCPU)
_GCU_QUANTIZED_KEY = "QuantizedPrivateUse1"
_GCU_PLAIN_KEY = "PrivateUse1"

# The Library must outlive this module: dropping the last reference unregisters
# every kernel it installed.
_GCU_QUANT_BRIDGE_LIBRARY = None


def _gcu_register(library, op_name, kernel, dispatch_key):
    """Install one bridge kernel; tolerate an occupied slot, stay fail-soft."""
    try:
        library.impl(op_name, kernel, dispatch_key, allow_override=True)
        return True
    except Exception as err:  # noqa: BLE001
        logger.debug(
            "GCU quantized bridge: %s at %s not registered: %s",
            op_name,
            dispatch_key,
            err,
        )
        return False


def _redispatch_quantized_cpu(op_name):
    """Build a kernel running the native ``QuantizedCPU`` implementation.

    Used for quantizer / TensorImpl metadata reads, which never touch device
    memory and therefore behave identically for a GCU argument.
    """
    native = getattr(torch.ops.aten, op_name).default

    def kernel(self):
        return native.redispatch(_QUANTIZED_CPU_KEYSET, self)

    return kernel


def _empty_quantized_gcu(
    size,
    qtensor,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    """``aten::empty_quantized``: allocate carrying ``qtensor``'s quantizer.

    The ``QuantizedCPU`` factories honour the ``device`` argument, so the
    redispatched native allocator runs exactly once on the requested device.
    """
    if qtensor.qscheme() in (
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ):
        native = torch.ops.aten._empty_per_channel_affine_quantized.default
        return native.redispatch(
            _QUANTIZED_CPU_KEYSET,
            list(size),
            scales=qtensor.q_per_channel_scales(),
            zero_points=qtensor.q_per_channel_zero_points(),
            axis=qtensor.q_per_channel_axis(),
            dtype=dtype if dtype is not None else qtensor.dtype,
            layout=layout,
            device=device if device is not None else qtensor.device,
            pin_memory=pin_memory,
            memory_format=memory_format,
        )
    native = torch.ops.aten._empty_affine_quantized.default
    return native.redispatch(
        _QUANTIZED_CPU_KEYSET,
        list(size),
        dtype=dtype if dtype is not None else qtensor.dtype,
        layout=layout,
        device=device if device is not None else qtensor.device,
        pin_memory=pin_memory,
        scale=qtensor.q_scale(),
        zero_point=qtensor.q_zero_point(),
        memory_format=(
            memory_format if memory_format is not None else torch.contiguous_format
        ),
    )


def _copy_from_gcu(self, dst, non_blocking=False):
    """``aten::_copy_from``: the quantized copy worker, over integer views.

    The raw integer representation is moved with the plain-tensor copy
    kernels torch_gcu provides (device-to-device and host-to-device alike),
    which is exactly the semantics of every call site in this operator's
    flow: both sides always carry the same quantizer, so no dequantize /
    requantize round trip is needed.  ``dst`` is resized to ``self``'s shape
    when they differ, keeping ATen's contract.
    """
    src = _int_view(self) if self.is_quantized else self
    dst_view = _int_view(dst) if dst.is_quantized else dst
    if tuple(dst_view.shape) != tuple(src.shape):
        strides = []
        acc = 1
        for dim in reversed(tuple(src.shape)):
            strides.append(acc)
            acc *= dim
        dst_view.set_(
            dst_view.untyped_storage(), 0, tuple(src.shape), tuple(reversed(strides))
        )
    dst_view.copy_(src, non_blocking=non_blocking)
    return dst


def _set_from_storage_offset_gcu(self, source, storage_offset, size, stride):
    """``aten::set_.source_Storage_storage_offset`` for a quantized tensor.

    Pure TensorImpl metadata surgery (storage swap + sizes/strides), so the
    native ``QuantizedCPU`` kernel handles a GCU tensor identically.
    """
    native = torch.ops.aten.set_.source_Storage_storage_offset
    return native.redispatch(
        _QUANTIZED_CPU_KEYSET, self, source, storage_offset, size, stride
    )


def _as_strided_gcu(self, size, stride, storage_offset=None):
    """``aten::as_strided`` for a quantized tensor: a pure metadata view.

    The native ``QuantizedCPU`` kernel only rebuilds the TensorImpl's sizes /
    strides / storage offset and never touches device memory, so it handles a
    GCU argument identically.  Needed by the view ops the layout tests use
    (``transpose`` lowers to ``as_strided``).
    """
    native = torch.ops.aten.as_strided.default
    return native.redispatch(
        _QUANTIZED_CPU_KEYSET, self, list(size), list(stride), storage_offset
    )


def _clone_gcu(self, memory_format=None):
    """``aten::clone`` for a quantized tensor.

    ``Tensor.contiguous(memory_format=...)`` lowers to ``clone``, whose
    composite then allocates through ``empty_like`` -> ``empty`` -- none of
    which have a kernel for a quantized GCU tensor.  The clone is built
    directly instead: the allocation reuses the ``QuantizedCPU`` factories
    (which honour the ``device`` argument and carry ``self``'s quantizer over
    to the result, matching native clone semantics) and the values are moved
    by the integer-view copy in :func:`_copy_from_gcu`.  ``preserve_format``
    mirrors ``empty_like``: channels-last stays channels-last, anything else
    allocates contiguous.
    """
    if memory_format is None or memory_format == torch.preserve_format:
        if self.dim() == 4 and self.is_contiguous(memory_format=torch.channels_last):
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
    if self.qscheme() in (
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ):
        out = torch.ops.aten._empty_per_channel_affine_quantized.default.redispatch(
            _QUANTIZED_CPU_KEYSET,
            list(self.shape),
            scales=self.q_per_channel_scales(),
            zero_points=self.q_per_channel_zero_points(),
            axis=self.q_per_channel_axis(),
            dtype=self.dtype,
            layout=None,
            device=self.device,
            pin_memory=None,
            memory_format=memory_format,
        )
    else:
        out = torch.ops.aten._empty_affine_quantized.default.redispatch(
            _QUANTIZED_CPU_KEYSET,
            list(self.shape),
            dtype=self.dtype,
            layout=None,
            device=self.device,
            pin_memory=None,
            scale=self.q_scale(),
            zero_point=self.q_zero_point(),
            memory_format=memory_format,
        )
    if self.numel() != 0:
        _copy_from_gcu(self, out)
    return out


def _quantize_per_tensor_gcu(a, scale, zero_point, dtype):
    """``aten::quantize_per_tensor`` over a plain GCU tensor.

    torch_gcu registers no kernel and the composite native exists only for
    CPU, so the quantization runs on the native CPU kernel over a host copy
    and the quantized result moves back to the input's device through the
    bridged quantized copy path.  Used by the benchmark's input construction
    (outside the timed region); the values match the CPU reference exactly
    because it *is* the CPU reference.
    """
    native = torch.ops.aten.quantize_per_tensor.default
    on_host = a.detach().to("cpu")
    quantized = native(on_host, scale, zero_point, dtype)
    return quantized.to(a.device)


def _quantize_per_channel_gcu(a, scales, zero_points, axis, dtype):
    """``aten::quantize_per_channel`` over plain GCU tensors.

    Same host-compute / device-move strategy as
    :func:`_quantize_per_tensor_gcu`; the native CPU kernel also owns the
    argument validation, so the error contract matches ATen verbatim.
    """
    native = torch.ops.aten.quantize_per_channel.default
    on_host = a.detach().to("cpu")
    quantized = native(
        on_host,
        scales.detach().to("cpu"),
        zero_points.detach().to("cpu"),
        axis,
        dtype,
    )
    return quantized.to(a.device)


def _install_uint8_min_shim():
    """Narrow ``Tensor.min`` shim: torch_gcu aborts on a Byte reduction.

    ``topsatenMin`` rejects ``uint8`` tensors with a hard ``Check failed``
    that kills the whole process (no catchable exception), while every other
    integer dtype works.  ``int_repr()`` of a ``quint8`` tensor is ``uint8``,
    and the accuracy suite takes ``result.int_repr().min().item()`` to check
    the padding sentinel, so the gate would abort after the kernel had
    already produced the right values.  The shim intercepts exactly that
    form -- a no-argument reduction over a GCU ``uint8`` tensor -- and
    computes it over an ``int32`` upcast (value-preserving: a uint8 minimum
    fits in uint8); every other call passes through to the stock method
    verbatim.  Same pattern as the complex-cast shim in
    ``view_as_complex_copy.py``.
    """
    if getattr(torch, "_flaggems_gcu_uint8_min_shim", False):
        return
    original_min = torch.Tensor.min

    def _uint8_safe_min(self, *args, **kwargs):
        if (
            not args
            and not kwargs
            and self.dtype == torch.uint8
            and self.device.type == "gcu"
        ):
            return original_min(self.to(torch.int32)).to(torch.uint8)
        return original_min(self, *args, **kwargs)

    torch.Tensor.min = _uint8_safe_min
    torch._flaggems_gcu_uint8_min_shim = True


def _install_gcu_quantized_bridge():
    """Register the quantized aten surface for GCU; idempotent and fail-soft."""
    global _GCU_QUANT_BRIDGE_LIBRARY
    if getattr(torch, "_flaggems_gcu_quantized_bridge", False):
        return
    try:
        from .dequantize import dequantize as _gcu_dequantize
        from .empty_affine_quantized import _empty_affine_quantized as _gcu_empty_affine
        from .empty_affine_quantized import (
            _empty_affine_quantized_out as _gcu_empty_affine_out,
        )
        from .empty_per_channel_affine_quantized import (
            _empty_per_channel_affine_quantized as _gcu_empty_per_channel,
        )
        from .int_repr import int_repr as _gcu_int_repr
    except Exception as err:  # noqa: BLE001
        logger.debug("GCU quantized bridge: implementation import failed: %s", err)
        return

    library = torch.library.Library("aten", "IMPL")
    _GCU_QUANT_BRIDGE_LIBRARY = library

    # Metadata readers: native QuantizedCPU kernels are device independent.
    for meta_op in (
        "qscheme",
        "q_scale",
        "q_zero_point",
        "q_per_channel_scales",
        "q_per_channel_zero_points",
        "q_per_channel_axis",
    ):
        _gcu_register(
            library, meta_op, _redispatch_quantized_cpu(meta_op), _GCU_QUANTIZED_KEY
        )

    # Factories and the quantized copy worker used by the native
    # ``_make_per_tensor_quantized_tensor`` chain and by the out= variant.
    _gcu_register(library, "empty_quantized", _empty_quantized_gcu, _GCU_QUANTIZED_KEY)
    _gcu_register(library, "_copy_from", _copy_from_gcu, _GCU_QUANTIZED_KEY)
    _gcu_register(
        library,
        "set_.source_Storage_storage_offset",
        _set_from_storage_offset_gcu,
        _GCU_QUANTIZED_KEY,
    )

    # Layout helpers: clone backs ``contiguous(memory_format=...)`` and
    # as_strided backs the view ops (transpose) the layout tests exercise.
    _gcu_register(library, "clone", _clone_gcu, _GCU_QUANTIZED_KEY)
    _gcu_register(library, "as_strided", _as_strided_gcu, _GCU_QUANTIZED_KEY)

    # FlagGems' own quantized factories, redispatching to QuantizedCPU.
    _gcu_register(
        library, "_empty_affine_quantized", _gcu_empty_affine, _GCU_QUANTIZED_KEY
    )
    _gcu_register(
        library,
        "_empty_affine_quantized.out",
        _gcu_empty_affine_out,
        _GCU_QUANTIZED_KEY,
    )
    _gcu_register(
        library,
        "_empty_per_channel_affine_quantized",
        _gcu_empty_per_channel,
        _GCU_QUANTIZED_KEY,
    )

    # Triton-implemented quantized surface from this backend.
    _gcu_register(library, "int_repr", _gcu_int_repr, _GCU_QUANTIZED_KEY)
    _gcu_register(library, "int_repr.out", _gcu_int_repr, _GCU_QUANTIZED_KEY)
    _gcu_register(library, "dequantize.self", _gcu_dequantize, _GCU_QUANTIZED_KEY)

    # The pooling operator itself, so a dispatched call reaches Triton.
    _gcu_register(
        library, "quantized_max_pool2d", quantized_max_pool2d, _GCU_QUANTIZED_KEY
    )
    _gcu_register(
        library,
        "quantized_max_pool2d.out",
        quantized_max_pool2d_out,
        _GCU_QUANTIZED_KEY,
    )

    # Quantize over plain GCU tensors (float inputs): torch_gcu registers no
    # kernel and the composite native only exists for CPU.  Needed by the
    # benchmark's input construction and the per-channel rejection tests.  The
    # backend's own Triton ``quantize_per_tensor`` / ``quantize_per_channel``
    # kernels do not compile on GCU400 (they compute in fp64, which the device
    # does not support), so the bridge routes through the native CPU kernel
    # and moves the quantized result back to the device instead.
    _gcu_register(
        library, "quantize_per_tensor", _quantize_per_tensor_gcu, _GCU_PLAIN_KEY
    )
    _gcu_register(
        library, "quantize_per_channel", _quantize_per_channel_gcu, _GCU_PLAIN_KEY
    )

    torch._flaggems_gcu_quantized_bridge = True


_install_gcu_quantized_bridge()
_install_uint8_min_shim()

# ---------------------------------------------------------------------------
# Generic-path module alias (vendor injection).
#
# SpecOpRegistrar imports this file as ``gcu400.ops.quantized_max_pool2d`` and
# replaces ``flag_gems.quantized_max_pool2d`` with the function above, but the
# accuracy suite resolves the implementation module through the generic path
# (``importlib.import_module("flag_gems.ops.quantized_max_pool2d")``) to
# monkeypatch ``quantized_max_pool2d_kernel`` in the launch-counting test, and
# the benchmark imports ``_parse_pool_params`` / ``max_pool2d_output_size``
# the same way.  Without this alias those patches land on the unused generic
# module while ``flag_gems.quantized_max_pool2d`` runs this one, and the
# instrumentation asserts fail even though the op is wired correctly.  Publish
# this module under the generic path so the module under test IS the
# implementation that runs.
# ---------------------------------------------------------------------------
if sys.modules.get("flag_gems.ops.quantized_max_pool2d") is not sys.modules.get(
    __name__
):
    sys.modules["flag_gems.ops.quantized_max_pool2d"] = sys.modules[__name__]
