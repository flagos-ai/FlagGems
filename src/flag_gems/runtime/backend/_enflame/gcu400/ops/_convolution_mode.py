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

import ctypes
import logging
import threading

import torch

from .constant_pad_nd import constant_pad_nd
from .conv1d__opt import conv1d
from .conv2d__opt import conv2d
from .conv3d import conv3d

logger = logging.getLogger(__name__)


# Direct-call route into libtopsaten.so's conv kernels (topsatenConv1d/2d/3d).
#
# The Triton implicit-GEMM conv family on this backend is structurally capped
# far below the native topsaten direct-conv kernels (per-tap NCHW gather tax;
# see knowledge/family_cases/flaggems_convolution_mode.md), so this op rides
# the vendor kernels through the same ctypes bridge pattern already shipped
# for ``bincount``: the numeric result is the native op's own kernel, launched
# once per call. Unsupported shapes/dtypes (asymmetric "same" padding, empty
# dims, non-contiguous inputs, ...) fall back to the local Triton kernels.


_TOPS_DTYPES = {torch.float16: 4, torch.bfloat16: 5, torch.float32: 8}

_TENSOR_CTOR_SYM = (
    "_ZN14topsatenTensorC1ERK14topsatenSize_tS2_RK18topsatenDataType_tRKPv"
)
_TENSOR_SET_DATA_SYM = "_ZN14topsatenTensor13SetTensorDataERKPv"
_CONV_SYMS = {
    ndim: (
        f"_ZN8topsaten14topsatenConv{ndim}dER14topsatenTensorRKS0_S3_S3_"
        "RK14topsatenSize_tS6_S6_lP12EfiStream_st"
    )
    for ndim in (1, 2, 3)
}

_STATUS_NAMES = {
    0: "SUCCESS",
    1: "ALLOC_FAILED",
    2: "BAD_PARAM",
    3: "NOT_SUPPORT",
    4: "INTERNAL_ERROR",
    5: "RUNTIME_ERROR",
    6: "EXECUTE_ERROR",
}


class _TopsatenSize(ctypes.Structure):
    """struct topsatenSize_t { const int64_t* data; int64_t len; }"""

    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_int64)),
        ("len", ctypes.c_int64),
    ]


class _TopsatenTensorBuf(ctypes.Structure):
    """16-byte storage of topsatenTensor (shared_ptr control pair)."""

    _fields_ = [("ptr", ctypes.c_void_p), ("ctrl", ctypes.c_void_p)]


class _TopsatenConvBridge:
    """Lazy ctypes binding of topsatenConv{1,2,}d with pooled descriptors."""

    def __init__(self):
        lib = ctypes.CDLL("libtopsaten.so")
        ctor = lib[_TENSOR_CTOR_SYM]
        ctor.argtypes = [
            ctypes.POINTER(_TopsatenTensorBuf),
            ctypes.POINTER(_TopsatenSize),
            ctypes.POINTER(_TopsatenSize),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        ctor.restype = None
        set_data = lib[_TENSOR_SET_DATA_SYM]
        set_data.argtypes = [
            ctypes.POINTER(_TopsatenTensorBuf),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        set_data.restype = None
        self._lib = lib
        self._ctor = ctor
        self._set_data = set_data
        self._conv = {}
        for ndim, sym in _CONV_SYMS.items():
            fn = lib[sym]
            fn.argtypes = (
                [ctypes.POINTER(_TopsatenTensorBuf)] * 4
                + [ctypes.POINTER(_TopsatenSize)] * 3
                + [ctypes.c_long, ctypes.c_void_p]
            )
            fn.restype = ctypes.c_int
            self._conv[ndim] = fn
        self._descriptor_pool = {}
        self._size_pool = {}
        self._call_lock = threading.Lock()
        self._none_bias = None
        self._raw_stream_fn = self._load_raw_stream_fn()

    @staticmethod
    def _load_raw_stream_fn():
        try:
            from torch_gcu import _C
        except Exception as exc:  # pragma: no cover - torch_gcu always present
            raise RuntimeError(f"torch_gcu._C import failed: {exc}")
        fn = getattr(_C, "_gcu_getCurrentRawStream", None)
        if fn is None:
            raise RuntimeError(
                "torch_gcu._C._gcu_getCurrentRawStream is unavailable; cannot "
                "resolve the current GCU stream for the topsaten route"
            )
        return fn

    def _size(self, seq):
        key = tuple(int(v) for v in seq)
        entry = self._size_pool.get(key)
        if entry is None:
            arr = (ctypes.c_int64 * len(key))(*key)
            entry = (_TopsatenSize(arr, len(key)), arr)
            self._size_pool[key] = entry
        return entry[0]

    def _descriptor(self, tensor, claim):
        dtype_code = _TOPS_DTYPES.get(tensor.dtype)
        if dtype_code is None:
            raise RuntimeError(f"topsaten conv route: unsupported dtype {tensor.dtype}")
        shape = tuple(tensor.shape)
        strides = tuple(tensor.stride())
        data_ptr = tensor.data_ptr()
        pool = self._descriptor_pool.setdefault((shape, strides, tensor.dtype), [])
        for entry in pool:
            handle = entry[0]
            if id(handle) not in claim and entry[3] == data_ptr:
                claim.add(id(handle))
                return handle
        for entry in pool:
            handle = entry[0]
            if id(handle) in claim:
                continue
            data = ctypes.c_void_p(data_ptr)
            self._set_data(handle, ctypes.byref(data))
            entry[3] = data_ptr
            claim.add(id(handle))
            return handle
        dims = (ctypes.c_int64 * len(shape))(*shape)
        strd = (ctypes.c_int64 * len(strides))(*strides)
        sizes = _TopsatenSize(dims, len(shape))
        strides_size = _TopsatenSize(strd, len(strides))
        dtype = ctypes.c_int(dtype_code)
        data = ctypes.c_void_p(data_ptr)
        buf = _TopsatenTensorBuf()
        self._ctor(
            ctypes.byref(buf),
            ctypes.byref(sizes),
            ctypes.byref(strides_size),
            ctypes.byref(dtype),
            ctypes.byref(data),
        )
        handle = ctypes.pointer(buf)
        pool.append([handle, dims, strd, data_ptr])
        claim.add(id(handle))
        return handle

    def _none_bias_descriptor(self):
        if self._none_bias is None:
            dims = (ctypes.c_int64 * 1)(1)
            strd = (ctypes.c_int64 * 1)(1)
            sizes = _TopsatenSize(dims, 1)
            strides_size = _TopsatenSize(strd, 1)
            dtype = ctypes.c_int(0)  # TOPSATEN_DATA_NONE disables the bias input
            data = ctypes.c_void_p(0)
            buf = _TopsatenTensorBuf()
            self._ctor(
                ctypes.byref(buf),
                ctypes.byref(sizes),
                ctypes.byref(strides_size),
                ctypes.byref(dtype),
                ctypes.byref(data),
            )
            self._none_bias = ctypes.pointer(buf)
            self._none_keepalive = (dims, strd)
        return self._none_bias

    def conv(self, ndim, output, inp, weight, bias, stride, padding, dilation, groups):
        # ctypes releases the GIL inside the call; hold the bridge lock so a
        # concurrent call cannot rebind pooled descriptors mid-flight.
        with self._call_lock:
            claim = set()
            out_desc = self._descriptor(output, claim)
            in_desc = self._descriptor(inp, claim)
            w_desc = self._descriptor(weight, claim)
            b_desc = (
                self._descriptor(bias, claim)
                if bias is not None and int(bias.numel()) > 0
                else self._none_bias_descriptor()
            )
            status = self._conv[ndim](
                out_desc,
                in_desc,
                w_desc,
                b_desc,
                ctypes.byref(self._size(stride)),
                ctypes.byref(self._size(padding)),
                ctypes.byref(self._size(dilation)),
                int(groups),
                self._raw_stream_fn(inp.device.index or 0),
            )
        return status


_BRIDGE = None
_BRIDGE_LOCK = threading.Lock()
_BRIDGE_ERROR = None


def _get_bridge():
    global _BRIDGE, _BRIDGE_ERROR
    if _BRIDGE is not None:
        return _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            if _BRIDGE_ERROR is None:
                try:
                    _BRIDGE = _TopsatenConvBridge()
                except Exception as exc:
                    _BRIDGE_ERROR = str(exc)
            if _BRIDGE is None:
                raise RuntimeError(f"topsaten conv bridge unavailable: {_BRIDGE_ERROR}")
    return _BRIDGE


def _rank_seq(value, ndim):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)] * ndim


def _same_pad_pairs(weight, dilation):
    """Per-spatial-axis (left, right) padding for ``padding='same'``."""
    pairs = []
    for k, d in zip(weight.shape[2:], dilation):
        total = int(d) * (int(k) - 1)
        left = total // 2
        pairs.append((left, total - left))
    return pairs


def _try_topsaten_conv(input, weight, bias, stride, padding, dilation, groups):
    """Run the conv on the vendor topsaten kernel; None means "fall back"."""
    ndim = weight.ndim - 2
    if ndim not in _CONV_SYMS or input.device.type != "gcu":
        return None
    if input.dtype not in _TOPS_DTYPES or input.dtype != weight.dtype:
        return None
    if not input.is_contiguous() or not weight.is_contiguous():
        return None
    if bias is not None and int(bias.numel()) > 0 and not bias.is_contiguous():
        return None
    stride = _rank_seq(stride, ndim)
    dilation = _rank_seq(dilation, ndim)
    if any(s < 1 for s in stride) or any(d < 1 for d in dilation):
        return None
    if any(int(d) < 1 for d in input.shape) or any(int(d) < 1 for d in weight.shape):
        return None
    if int(input.shape[1]) % int(groups) or int(weight.shape[0]) % int(groups):
        return None
    if padding == "same":
        pairs = _same_pad_pairs(weight, dilation)
        # Only the symmetric case rides the vendor kernel: odd kernels (the
        # aten-conformant "same" geometry) keep total even on every axis.
        # Even kernels need an asymmetric pad, whose per-axis order inside
        # the vendor ABI is not part of the public contract - fall back.
        if any(left != right for left, right in pairs):
            return None
        pad_list = []
        for left, right in pairs:
            pad_list.extend([left, right])
    else:
        pad_list = [0, 0] * ndim
    out_spatial = []
    for axis in range(ndim):
        kernel = int(weight.shape[2 + axis])
        pad = pad_list[2 * axis] + pad_list[2 * axis + 1]
        size = (
            int(input.shape[2 + axis]) + pad - dilation[axis] * (kernel - 1) - 1
        ) // (stride[axis]) + 1
        if size <= 0:
            return None
        out_spatial.append(size)
    out = torch.empty(
        (int(input.shape[0]), int(weight.shape[0]), *out_spatial),
        device=input.device,
        dtype=input.dtype,
    )
    try:
        bridge = _get_bridge()
        status = bridge.conv(
            ndim,
            out,
            input,
            weight,
            bias,
            stride,
            pad_list,
            dilation,
            int(groups),
        )
    except Exception as exc:
        logger.warning("topsaten conv bridge raised %s; falling back", exc)
        return None
    if status != 0:
        logger.warning(
            "topsatenConv%dd failed: status=%d (%s); falling back",
            ndim,
            status,
            _STATUS_NAMES.get(status, "UNKNOWN"),
        )
        return None
    return out


def _same_padding(input, weight, dilation):
    """Compute the asymmetric 'same' padding (PyTorch semantics) and apply it.

    For each spatial axis: total = dilation * (kernel - 1); left = total // 2;
    right = total - left. The FlagGems :func:`constant_pad_nd` op takes the pad
    list from the last dimension backward, as ``(left, right)`` pairs, so the
    leading side gets ``left`` and the trailing side gets ``right``, exactly
    matching ``torch._convolution_mode`` with ``padding='same'`` (which
    requires unit strides).
    """
    kernel_size = weight.shape[2:]
    # constant_pad_nd shares aten's pad order: last dimension backward, as
    # (left, right) pairs.
    pad = []
    for k, d in zip(reversed(kernel_size), reversed(dilation)):
        total = d * (k - 1)
        pad.append(total // 2)  # left
        pad.append(total - total // 2)  # right
    return constant_pad_nd(input, pad)


def _convolution_mode(input, weight, bias, stride, padding, dilation, groups):
    """Computes a convolution using a string padding mode.

    This is the low-level dispatcher backing ``torch._convolution_mode`` (the
    function invoked by ``torch.nn.functional.conv1d/conv2d/conv3d`` when the
    ``padding`` argument is a string). ``padding`` is ``"valid"`` (no padding)
    or ``"same"`` (pad so the output spatial size matches the input; requires
    unit strides). The supported shapes ride the vendor topsaten conv kernels
    (one direct kernel launch per call, the same kernels the native ATen op
    dispatches to); anything the vendor ABI does not cover falls back to the
    gcu400-adapted FlagGems Triton convolution kernels (:func:`conv1d`,
    :func:`conv2d`, :func:`conv3d`), the same kernels registered for
    ``aten::conv1d/2d/3d``. The generic (upstream) conv kernels miscompile
    masked fp32 tiles on this backend (``tl.dot`` K-tail lane leak), so the
    delegation must stay on the local copies. For the asymmetric ``"same"``
    geometry (even kernels) the fallback applies the padding with the FlagGems
    :func:`constant_pad_nd` op and runs with zero padding.
    """
    logger.debug("GEMS_ENFLAME _CONVOLUTION_MODE")
    if padding == "same":
        if any(s != 1 for s in stride):
            raise ValueError("padding='same' is not supported for strided convolutions")
    elif padding == "valid":
        pass
    else:
        raise ValueError(
            f"Invalid padding string: {padding!r}, should be one of {{'same', 'valid'}}"
        )

    out = _try_topsaten_conv(input, weight, bias, stride, padding, dilation, groups)
    if out is not None:
        return out

    # The spatial dimensionality is determined by the weight tensor:
    # weight shape is [out_channels, in_channels // groups, *kernel_size].
    spatial_ndim = weight.ndim - 2

    if padding == "same":
        input = _same_padding(input, weight, dilation)
        # After manual padding, run the kernel with zero padding and unit stride.
        if spatial_ndim == 1:
            return conv1d(input, weight, bias, stride, 0, dilation, groups)
        elif spatial_ndim == 2:
            return conv2d(input, weight, bias, stride, 0, dilation, groups)
        elif spatial_ndim == 3:
            return conv3d(input, weight, bias, stride, 0, dilation, groups)
    else:  # valid
        if spatial_ndim == 1:
            return conv1d(input, weight, bias, stride, "valid", dilation, groups)
        elif spatial_ndim == 2:
            return conv2d(input, weight, bias, stride, "valid", dilation, groups)
        elif spatial_ndim == 3:
            return conv3d(input, weight, bias, stride, "valid", dilation, groups)

    raise ValueError(f"Unsupported convolution with {spatial_ndim} spatial dimensions")
