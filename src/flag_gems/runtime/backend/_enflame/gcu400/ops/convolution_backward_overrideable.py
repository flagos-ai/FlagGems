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

"""aten::convolution_backward_overrideable (2D, non-transposed) on GCU400.

Supported geometries first ride the vendor topsaten conv-backward kernel
(``topsatenConvolutionBackward``) through the ctypes bridge pattern already
shipped for ``_convolution_mode``/``bincount``: the numeric result is the
native op's own kernel, launched once per call (the Triton implicit-GEMM
conv family on this backend is structurally capped at 2-4 TF/s by the
per-tap NCHW gather tax, while the vendor backward kernels reach ~12.5
TF/s on fp16).  Anything the vendor ABI does not accept falls back to the
GCU400 Triton fast path below; the original composable path
(conv_transpose2d + the generic weight-gradient kernel + sum_dim) stays as
the final fallback (fp64, non-contiguous inputs, tiny per-group channel
counts, ...).

  * grad_input  : residue-phase implicit GEMM.  grad_input is the transposed
    convolution of grad_output by the weight; output positions are partitioned
    by their (hi % sh, wi % sw) phase so each program only visits the taps
    whose back-projection lands on a grad_output sample (the structure the
    GCU-adapted conv_transpose kernels use).  fp16/bf16 upcast to fp32 before
    the ieee dot: native low-precision dots are intermittently wrong on this
    backend, while the fp32 ieee dot is deterministic.
  * grad_weight : split-K implicit GEMM over the flattened (n, ho) reduction
    axis with the (ci, kh, kw) output axis folded into tile columns; the
    per-(n, ho) work chunk keeps the wo axis a plain affine arange so the dot
    operand loads never degrade to per-element gathers.  Partial sums land in
    an fp32 slab and one reduce kernel folds the split axis (and the bias
    partials that ride along on the same slabs) into the outputs.
  * grad_bias   : folds onto the grad_weight split-K slabs for free when both
    are requested; a standalone reduction handles the remaining masks.
"""

import ctypes
import logging
import threading

import torch
import triton
import triton.language as tl

from flag_gems.ops.conv2d import conv2d_backward_kernel_weight
from flag_gems.ops.sum import sum_dim as gems_sum_dim
from flag_gems.utils import libentry

from .conv_transpose2d import conv_transpose2d

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.float32, torch.bfloat16)

# input_precision for the fp32-ieee/tf32 dot variants of the upcast half-
# precision path (A/B knob; "ieee" is the deterministic baseline).
_HALF_DOT_PREC = "tf32"


# --- Vendor topsaten conv-backward direct-call route ----------------------
#
# ABI facts for ``topsaten::topsatenConvolutionBackward`` (verified against
# the native torch_gcu call site in gdb, 2026-11-18):
#   * the leading ``std::tuple<topsatenTensor&, ...>`` is passed as an
#     invisible reference (rdi -> 24-byte struct), not by value;
#   * libstdc++ stores tuple elements in REVERSE memory order, so the
#     24-byte struct is [get<2>=grad_bias, get<1>=grad_weight,
#     get<0>=grad_input];
#   * ``std::array<bool, 3>`` (output_mask) is passed by value as 3 bytes in
#     one stack slot;
#   * masked-off outputs are left untouched by the vendor kernel (sentinel
#     probe), matching aten semantics for the ``.out`` variant.

_TOPS_DTYPES = {torch.float16: 4, torch.bfloat16: 5, torch.float32: 8}

_TENSOR_CTOR_SYM = (
    "_ZN14topsatenTensorC1ERK14topsatenSize_tS2_RK18topsatenDataType_tRKPv"
)
_TENSOR_SET_DATA_SYM = "_ZN14topsatenTensor13SetTensorDataERKPv"
_CONV_BWD_SYM = (
    "_ZN8topsaten27topsatenConvolutionBackwardESt5tupleIJR14topsatenTensorS2_S2_"
    "EERKS1_S5_S5_RK14topsatenSize_tS8_S8_S8_bS8_lSt5arrayIbLm3EEP12EfiStream_st"
)

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


class _ConvBwdOutTuple(ctypes.Structure):
    """std::tuple<topsatenTensor&, topsatenTensor&, topsatenTensor&> (24B).

    Memory order is [get<2>, get<1>, get<0>] = (grad_bias, grad_weight,
    grad_input); see the ABI notes above.
    """

    _fields_ = [
        ("g2", ctypes.POINTER(_TopsatenTensorBuf)),  # std::get<2>: grad_bias
        ("g1", ctypes.POINTER(_TopsatenTensorBuf)),  # std::get<1>: grad_weight
        ("g0", ctypes.POINTER(_TopsatenTensorBuf)),  # std::get<0>: grad_input
    ]


class _Bool3(ctypes.Structure):
    """std::array<bool, 3> (3-byte by-value stack slot)."""

    _fields_ = [
        ("b0", ctypes.c_bool),
        ("b1", ctypes.c_bool),
        ("b2", ctypes.c_bool),
    ]


class _TopsatenConvBackwardBridge:
    """Lazy ctypes binding of topsatenConvolutionBackward with pooled
    descriptors (same lifetime/anti-aliasing discipline as the
    ``_convolution_mode`` bridge)."""

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
        conv_bwd = lib[_CONV_BWD_SYM]
        conv_bwd.argtypes = [
            ctypes.POINTER(_ConvBwdOutTuple),  # outputs (invisible ref)
            ctypes.POINTER(_TopsatenTensorBuf),  # grad_output
            ctypes.POINTER(_TopsatenTensorBuf),  # input
            ctypes.POINTER(_TopsatenTensorBuf),  # filter
            ctypes.POINTER(_TopsatenSize),  # bias_sizes
            ctypes.POINTER(_TopsatenSize),  # stride
            ctypes.POINTER(_TopsatenSize),  # padding
            ctypes.POINTER(_TopsatenSize),  # dilation
            ctypes.c_bool,  # transposed
            ctypes.POINTER(_TopsatenSize),  # output_padding
            ctypes.c_long,  # groups
            _Bool3,  # output_mask (by value)
            ctypes.c_void_p,  # stream
        ]
        conv_bwd.restype = ctypes.c_int
        self._lib = lib
        self._ctor = ctor
        self._set_data = set_data
        self._conv_bwd = conv_bwd
        self._descriptor_pool = {}
        self._size_pool = {}
        self._call_lock = threading.Lock()
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
            raise RuntimeError(
                f"topsaten conv-backward route: unsupported dtype {tensor.dtype}"
            )
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

    def conv_bwd(
        self,
        grad_output,
        input,
        weight,
        grad_input,
        grad_weight,
        grad_bias,
        stride,
        padding,
        dilation,
        groups,
        mask,
    ):
        # ctypes releases the GIL inside the call; hold the bridge lock so a
        # concurrent call cannot rebind pooled descriptors mid-flight.
        with self._call_lock:
            claim = set()
            go_desc = self._descriptor(grad_output, claim)
            in_desc = self._descriptor(input, claim)
            w_desc = self._descriptor(weight, claim)
            gi_desc = self._descriptor(grad_input, claim)
            gw_desc = self._descriptor(grad_weight, claim)
            gb_desc = self._descriptor(grad_bias, claim)
            outs = _ConvBwdOutTuple(gb_desc, gw_desc, gi_desc)
            mask3 = _Bool3(bool(mask[0]), bool(mask[1]), bool(mask[2]))
            status = self._conv_bwd(
                ctypes.byref(outs),
                go_desc,
                in_desc,
                w_desc,
                ctypes.byref(self._size([weight.shape[0]])),
                ctypes.byref(self._size(stride)),
                ctypes.byref(self._size(padding)),
                ctypes.byref(self._size(dilation)),
                False,
                ctypes.byref(self._size([0, 0])),
                int(groups),
                mask3,
                self._raw_stream_fn(input.device.index or 0),
            )
        return status


_CONV_BWD_BRIDGE = None
_CONV_BWD_BRIDGE_LOCK = threading.Lock()
_CONV_BWD_BRIDGE_ERROR = None


def _get_conv_bwd_bridge():
    global _CONV_BWD_BRIDGE, _CONV_BWD_BRIDGE_ERROR
    if _CONV_BWD_BRIDGE is not None:
        return _CONV_BWD_BRIDGE
    with _CONV_BWD_BRIDGE_LOCK:
        if _CONV_BWD_BRIDGE is None:
            if _CONV_BWD_BRIDGE_ERROR is None:
                try:
                    _CONV_BWD_BRIDGE = _TopsatenConvBackwardBridge()
                except Exception as exc:
                    _CONV_BWD_BRIDGE_ERROR = str(exc)
            if _CONV_BWD_BRIDGE is None:
                raise RuntimeError(
                    f"topsaten conv-backward bridge unavailable: {_CONV_BWD_BRIDGE_ERROR}"
                )
    return _CONV_BWD_BRIDGE


def _try_topsaten_conv_backward(
    grad_output,
    input,
    weight,
    stride,
    padding,
    dilation,
    groups,
    output_mask,
    out0,
    out1,
    out2,
):
    """Run the backward on the vendor topsaten kernel; None = "fall back".

    The vendor ABI covers fp16/bf16/fp32 contiguous NCHW tensors of any
    positive geometry and leaves masked-off outputs untouched (sentinel
    probe, 2026-11-18); masked-off slots still receive a properly-shaped
    buffer so the vendor-side parameter checks always see valid tensors.
    """
    if input.device.type != "gcu" or weight.ndim != 4 or input.ndim != 4:
        return None
    dtype = input.dtype
    if dtype not in _TOPS_DTYPES:
        return None
    if grad_output.dtype != dtype or weight.dtype != dtype:
        return None
    if not (
        grad_output.is_contiguous() and input.is_contiguous() and weight.is_contiguous()
    ):
        return None
    if (
        any(int(d) <= 0 for d in input.shape)
        or any(int(d) <= 0 for d in weight.shape)
        or any(int(d) <= 0 for d in grad_output.shape)
    ):
        return None
    if any(s < 1 for s in stride) or any(d < 1 for d in dilation):
        return None

    grad_input = out0 if output_mask[0] else None
    grad_weight = out1 if output_mask[1] else None
    grad_bias = out2 if output_mask[2] else None
    if output_mask[0]:
        if out0 is None:
            grad_input = torch.empty_like(input)
        elif (
            out0.dtype != dtype
            or tuple(out0.shape) != tuple(input.shape)
            or not out0.is_contiguous()
        ):
            return None
    else:
        grad_input = torch.empty_like(input)
    if output_mask[1]:
        if out1 is None:
            grad_weight = torch.empty_like(weight)
        elif (
            out1.dtype != dtype
            or tuple(out1.shape) != tuple(weight.shape)
            or not out1.is_contiguous()
        ):
            return None
    else:
        grad_weight = torch.empty_like(weight)
    bias_shape = (weight.shape[0],)
    if output_mask[2]:
        if out2 is None:
            grad_bias = torch.empty(bias_shape, dtype=dtype, device=input.device)
        elif (
            out2.dtype != dtype
            or tuple(out2.shape) != bias_shape
            or not out2.is_contiguous()
        ):
            return None
    else:
        grad_bias = torch.empty(bias_shape, dtype=dtype, device=input.device)

    try:
        bridge = _get_conv_bwd_bridge()
        status = bridge.conv_bwd(
            grad_output,
            input,
            weight,
            grad_input,
            grad_weight,
            grad_bias,
            stride,
            padding,
            dilation,
            groups,
            output_mask,
        )
    except Exception as exc:
        logger.warning("topsaten conv-backward bridge raised %s; falling back", exc)
        return None
    if status != 0:
        logger.warning(
            "topsatenConvolutionBackward failed: status=%d (%s); falling back",
            status,
            _STATUS_NAMES.get(status, "UNKNOWN"),
        )
        return None
    return (
        grad_input if output_mask[0] else None,
        grad_weight if output_mask[1] else None,
        grad_bias if output_mask[2] else None,
    )


@libentry()
@triton.jit
def _cbo_dgrad_dot(
    go_ptr,
    w_ptr,
    gi_ptr,
    goff,
    woff,
    gioff,
    sgo_n,
    sgi_n,
    N: tl.constexpr,
    CO: tl.constexpr,
    HO: tl.constexpr,
    WO: tl.constexpr,
    CI: tl.constexpr,
    HI: tl.constexpr,
    WI: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    KH0: tl.constexpr,
    KW0: tl.constexpr,
    LIVEH: tl.constexpr,
    LIVEW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    RESH: tl.constexpr,
    RESW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    UPCAST: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # grad_input[n, ci, hi, wi] = sum_{co, kh, kw} grad_output[n, co, oh, ow]
    #   * w[co, ci, kh, kw], oh = (hi + PH - DH*kh) // SH, ow likewise.
    # One launch per (hi % SH, wi % SW) phase: RESH/RESW and the live tap
    # subsequence (KH0 + SH*th) are constexpr, so every tap's back-projection
    # is an exact integer offset OFF = (RESH + PH - kh*DH) // SH and the body
    # has no runtime control flow at all (runtime ifs around dots degrade to
    # scalar code on this backend).  Loads are masked and then re-zeroed with
    # tl.where right before the dot -- the fast-lowering recipe measured on
    # slow_conv2d_forward's v2 kernel.
    pid_p = tl.program_id(0)
    pid_co = tl.program_id(1)

    compact_h: tl.constexpr = (HI + SH - 1) // SH
    compact_w: tl.constexpr = (WI + SW - 1) // SW
    compact_plane: tl.constexpr = compact_h * compact_w
    total: tl.constexpr = N * compact_plane

    idx = pid_p * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = idx < total
    cw = idx % compact_w
    idx_h = idx // compact_w
    ch = idx_h % compact_h
    n = idx_h // compact_h
    hi = ch * SH + RESH
    wi = cw * SW + RESW

    ci = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    vci = ci < CI
    cic0 = tl.where(vci, ci, 0)

    acc = tl.zeros((BLOCK_N, BLOCK_CO), dtype=tl.float32)

    SGOC: tl.constexpr = HO * WO
    WKC: tl.constexpr = CI * KH * KW
    ci_blocks: tl.constexpr = (CO + BLOCK_CI - 1) // BLOCK_CI
    # The masked+where recipe lowers to unmasked loads + select, so every
    # lane's address must stay inside the real allocation, masked or not.
    nc = tl.minimum(n, N - 1)
    hic = tl.minimum(hi, HI - 1)
    wic = tl.minimum(wi, WI - 1)
    for th in tl.static_range(0, LIVEH):
        kh = KH0 + SH * th
        offh = (RESH + PH - kh * DH) // SH
        oh = ch + offh
        vd = valid & (oh >= 0) & (oh < HO)
        ohc = tl.minimum(tl.maximum(oh, 0), HO - 1)
        for tw in tl.static_range(0, LIVEW):
            kw = KW0 + SW * tw
            offw = (RESW + PW - kw * DW) // SW
            ow = cw + offw
            vm = vd & (ow >= 0) & (ow < WO)
            owc = tl.minimum(tl.maximum(ow, 0), WO - 1)
            go_row = nc * sgo_n + ohc * WO + owc
            for cb in tl.static_range(0, ci_blocks):
                kk = cb * BLOCK_CI + tl.arange(0, BLOCK_CI)
                vkk = kk < CO
                kkc = tl.where(vkk, kk, 0)
                cic = tl.where(vci, ci, 0)
                m2 = vm[:, None] & vkk[None, :]
                a = tl.load(
                    go_ptr + goff + go_row[:, None] + (kkc * SGOC)[None, :],
                    mask=m2,
                    other=0.0,
                )
                a = tl.where(m2, a, 0.0)
                mb = vkk[:, None] & vci[None, :]
                b = tl.load(
                    w_ptr
                    + woff
                    + (kkc * WKC)[:, None]
                    + ((cic * KH + kh) * KW + kw)[None, :],
                    mask=mb,
                    other=0.0,
                )
                b = tl.where(mb, b, 0.0)
                if UPCAST:
                    acc = acc + tl.dot(
                        a.to(tl.float32), b.to(tl.float32), input_precision="ieee"
                    )
                else:
                    acc = tl.dot(a, b, acc, input_precision="ieee")

    giptr = (
        gi_ptr
        + gioff
        + (nc * sgi_n)[:, None]
        + (cic0 * (HI * WI))[None, :]
        + hic[:, None] * WI
        + wic[:, None]
    )
    tl.store(
        giptr,
        acc.to(gi_ptr.dtype.element_ty),
        mask=(n[:, None] < N) & (hi[:, None] < HI) & (wi[:, None] < WI) & vci[None, :],
    )


@libentry()
@triton.jit
def _cbo_wgrad_dot(
    go_ptr,
    x_ptr,
    part_ptr,
    bpart_ptr,
    goff,
    xoff,
    poff,
    bpoff,
    sgo_n,
    sx_n,
    NH_TOTAL,
    CO: tl.constexpr,
    CI: tl.constexpr,
    N: tl.constexpr,
    HO: tl.constexpr,
    WO: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    HAS_B: tl.constexpr,
    UPCAST: tl.constexpr,
    NHPC: tl.constexpr,
    W_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # grad_weight[co, ci, kh, kw] = sum over (n, ho, wo) of
    #   grad_output[n, co, ho, wo] * input[n, ci, SH*ho + DH*kh - PH,
    #                                   SW*wo + DH*kw - PW].
    # The (n, ho) reduction axis is split across grid z and statically
    # unrolled NHPC rows per program (a loop-carried fp32 dot accumulator in
    # a dynamic range loop miscompiles on this backend); the wo chunk stays a
    # plain affine arange so the dot operand loads keep the fast lowering.
    # The (ci, kh, kw) output axis is folded into tile columns, so one dot
    # per chunk covers every tap.  Loads are masked and re-zeroed with
    # tl.where before the dot (fast-lowering recipe).  Full tensors are
    # passed with per-group element offsets and true batch strides.
    pid_k = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_m = tl.program_id(2)

    KK: tl.constexpr = KH * KW
    CIKK: tl.constexpr = CI * KK

    co = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mco = co < CO
    coc = tl.where(mco, co, 0)
    jj = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    mj = jj < CIKK
    jjc = tl.where(mj, jj, 0)
    cij = jjc // KK
    rj = jjc - cij * KK
    khj = rj // KW
    kwj = rj - khj * KW

    acc = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)
    if HAS_B:
        bsum = tl.zeros((BLOCK_M, BLOCK_W), dtype=tl.float32)
        bsel = pid_j == 0

    cij_c = tl.where(mj, cij, 0)
    wov = tl.arange(0, BLOCK_W)

    nh0 = pid_k * NHPC
    if W_CHUNKS == 1:
        # Single wo chunk: the wo-side tap window is loop-invariant, so it
        # is derived once instead of once per unrolled (n, ho) row.
        woffs = wov
        mwf = woffs < WO
        woffs_c = tl.where(mwf, woffs, 0)
        iw = SW * woffs_c[:, None] + (DW * kwj - PW)[None, :]
        okw = (iw >= 0) & (iw < W)
        iwc = tl.minimum(tl.maximum(iw, 0), W - 1)
        for i in tl.static_range(0, NHPC):
            nh = nh0 + i
            live = nh < NH_TOTAL
            n = nh // HO
            ho = nh - n * HO
            nc = tl.minimum(n, N - 1)
            hoc = tl.minimum(ho, HO - 1)
            go_row = goff + nc * sgo_n + hoc * WO
            m2 = live & mco[:, None] & mwf[None, :]
            a = tl.load(
                go_ptr + go_row + (coc * (HO * WO))[:, None] + woffs_c[None, :],
                mask=m2,
                other=0.0,
            )
            a = tl.where(m2, a, 0.0)
            if HAS_B:
                bsum += a.to(tl.float32)
            ih = SH * hoc + DH * khj - PH
            okh = live & (ih >= 0) & (ih < H) & mj
            ihc = tl.minimum(tl.maximum(ih, 0), H - 1)
            mb = mwf[:, None] & okh[None, :] & okw
            b = tl.load(
                x_ptr + xoff + nc * sx_n + (cij_c * (H * W) + ihc * W)[None, :] + iwc,
                mask=mb,
                other=0.0,
            )
            b = tl.where(mb, b, 0.0)
            if UPCAST:
                acc = acc + tl.dot(
                    a.to(tl.float32), b.to(tl.float32), input_precision="ieee"
                )
            else:
                acc = tl.dot(a, b, acc, input_precision="ieee")
    else:
        for i in tl.static_range(0, NHPC):
            nh = nh0 + i
            live = nh < NH_TOTAL
            n = nh // HO
            ho = nh - n * HO
            nc = tl.minimum(n, N - 1)
            hoc = tl.minimum(ho, HO - 1)
            go_row = goff + nc * sgo_n + hoc * WO
            for w0 in tl.static_range(0, W_CHUNKS):
                woffs = w0 * BLOCK_W + wov
                mwf = woffs < WO
                woffs_c = tl.where(mwf, woffs, 0)
                m2 = live & mco[:, None] & mwf[None, :]
                a = tl.load(
                    go_ptr + go_row + (coc * (HO * WO))[:, None] + woffs_c[None, :],
                    mask=m2,
                    other=0.0,
                )
                a = tl.where(m2, a, 0.0)
                if HAS_B:
                    bsum += a.to(tl.float32)
                ih = SH * hoc + DH * khj - PH
                okh = live & (ih >= 0) & (ih < H) & mj
                ihc = tl.minimum(tl.maximum(ih, 0), H - 1)
                iw = SW * woffs_c[:, None] + (DW * kwj - PW)[None, :]
                okw = (iw >= 0) & (iw < W)
                iwc = tl.minimum(tl.maximum(iw, 0), W - 1)
                mb = mwf[:, None] & okh[None, :] & okw
                b = tl.load(
                    x_ptr
                    + xoff
                    + nc * sx_n
                    + (cij_c * (H * W) + ihc * W)[None, :]
                    + iwc,
                    mask=mb,
                    other=0.0,
                )
                b = tl.where(mb, b, 0.0)
                if UPCAST:
                    acc = acc + tl.dot(
                        a.to(tl.float32),
                        b.to(tl.float32),
                        input_precision="ieee",
                    )
                else:
                    acc = tl.dot(a, b, acc, input_precision="ieee")

    part_base = poff + pid_k * (CO * CIKK) + coc[:, None] * CIKK + jjc[None, :]
    tl.store(part_ptr + part_base, acc, mask=mco[:, None] & mj[None, :])
    if HAS_B:
        bid = tl.sum(bsum, axis=1)
        tl.store(bpart_ptr + bpoff + pid_k * CO + coc, bid, mask=mco & bsel)


@libentry()
@triton.jit
def _cbo_wgrad_reduce(
    part_ptr,
    bpart_ptr,
    gw_ptr,
    gb_ptr,
    poff,
    bpoff,
    gwoff,
    gboff,
    CO: tl.constexpr,
    CI: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HAS_B: tl.constexpr,
    BLOCK_FLAT: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    # Fold the split-K fp32 slabs into grad_weight (and the riding bias
    # partials into grad_bias), casting once on the store.
    KK: tl.constexpr = KH * KW
    CIKK: tl.constexpr = CI * KK
    pid = tl.program_id(0)

    off = pid * BLOCK_FLAT + tl.arange(0, BLOCK_FLAT)
    mo = off < CO * CIKK
    offc = tl.where(mo, off, 0)
    # SPLIT_K can exceed the lane budget (BLOCK_S), so the slab axis is
    # folded in chunks; each chunk's addresses stay clamped in-range.
    acc = tl.zeros((BLOCK_FLAT,), dtype=tl.float32)
    for s0 in range(0, SPLIT_K, BLOCK_S):
        ss = s0 + tl.arange(0, BLOCK_S)
        ms = ss < SPLIT_K
        ssc = tl.where(ms, ss, 0)
        v = tl.load(
            part_ptr + poff + ssc[:, None] * (CO * CIKK) + offc[None, :],
            mask=ms[:, None] & mo[None, :],
            other=0.0,
        )
        acc += tl.sum(v.to(tl.float32), axis=0)

    co = offc // CIKK
    r = offc - co * CIKK
    cij = r // KK
    r2 = r - cij * KK
    khj = r2 // KW
    kwj = r2 - khj * KW
    tl.store(
        gw_ptr + gwoff + (co * CI + cij) * KK + khj * KW + kwj,
        acc.to(gw_ptr.dtype.element_ty),
        mask=mo,
    )

    if HAS_B:
        if pid == 0:
            for c0 in range(0, CO, BLOCK_B):
                cb = c0 + tl.arange(0, BLOCK_B)
                mcb = cb < CO
                cbc = tl.where(mcb, cb, 0)
                bacc = tl.zeros((BLOCK_B,), dtype=tl.float32)
                for b0 in range(0, SPLIT_K, BLOCK_S):
                    ss = b0 + tl.arange(0, BLOCK_S)
                    ms = ss < SPLIT_K
                    ssc = tl.where(ms, ss, 0)
                    vb = tl.load(
                        bpart_ptr + bpoff + ssc[:, None] * CO + cbc[None, :],
                        mask=ms[:, None] & mcb[None, :],
                        other=0.0,
                    )
                    bacc += tl.sum(vb.to(tl.float32), axis=0)
                tl.store(
                    gb_ptr + gboff + cbc,
                    bacc.to(gb_ptr.dtype.element_ty),
                    mask=mcb,
                )


# ---------------------------------------------------------------------------
# host side: tile selection and fast-path launchers
# ---------------------------------------------------------------------------
def _next_pow2(value):
    out = 1
    while out < value:
        out *= 2
    return out


def _int32_safe(*tensors):
    # Byte offsets must stay below 2**31 so the kernels' int32 address math
    # cannot wrap around (see the strided-2**31 compiler trap).
    for t in tensors:
        if t is not None and t.numel() * t.element_size() >= 2**31:
            return False
    return True


def _dgrad_tiles(dtype, co, ci, compact_total):
    """(BLOCK_N, BLOCK_CI, BLOCK_CO, num_warps, num_stages) for the dgrad.

    GCU400 sweep winners over the benchmark matrix: wide 256-position tiles
    with a 16-lane reduction tile win on the big planes (num_warps=1 at the
    very large sizes), 128-position tiles with 32-lane blocks win on the
    small ones.
    """
    block_ci = 16 if compact_total >= 16384 else max(16, min(32, _next_pow2(co)))
    block_co = max(16, min(64, _next_pow2(ci)))
    if compact_total >= 65536:
        return 256, block_ci, block_co, 1, 2
    if compact_total >= 16384:
        return 256, block_ci, block_co, 2, 2
    return 128, block_ci, block_co, 4, 2


def _phase_taps(k_size, step, dilation, residue, pad):
    """Live tap subsequence for one phase: (start, count) or None."""
    target = (residue + pad) % step
    taps = [k for k in range(k_size) if (k * dilation) % step == target]
    if not taps:
        return None
    return taps[0], len(taps)


def _dgrad_fast(grad_output, weight, grad_input, groups, sh, sw, ph, pw, dh, dw):
    """Launch the per-phase implicit-GEMM dgrad kernel for every group."""
    n = grad_output.shape[0]
    co_total = grad_output.shape[1]
    co_g = co_total // groups
    ci_g = weight.shape[1]
    kh, kw = weight.shape[2], weight.shape[3]
    ho, wo = grad_output.shape[2], grad_output.shape[3]
    hi, wi = grad_input.shape[2], grad_input.shape[3]
    sgo_n = co_total * ho * wo
    sgi_n = grad_input.shape[1] * hi * wi

    bn, bci, bco, warps, stages = _dgrad_tiles(
        grad_output.dtype,
        co_g,
        ci_g,
        n * ((hi + sh - 1) // sh) * ((wi + sw - 1) // sw),
    )
    upcast = grad_output.dtype != torch.float32
    compact_h = (hi + sh - 1) // sh
    compact_w = (wi + sw - 1) // sw
    grid0 = (
        triton.cdiv(n * compact_h * compact_w, bn),
        triton.cdiv(ci_g, bco),
    )
    for g in range(groups):
        goff = g * co_g * ho * wo
        woff = g * co_g * ci_g * kh * kw
        gioff = g * ci_g * hi * wi
        for resh in range(sh):
            tapsh = _phase_taps(kh, sh, dh, resh, ph)
            if tapsh is None:
                continue
            kh0, liveh = tapsh
            for resw in range(sw):
                tapsw = _phase_taps(kw, sw, dw, resw, pw)
                if tapsw is None:
                    continue
                kw0, livew = tapsw
                _cbo_dgrad_dot[grid0](
                    grad_output,
                    weight,
                    grad_input,
                    goff,
                    woff,
                    gioff,
                    sgo_n,
                    sgi_n,
                    n,
                    co_g,
                    ho,
                    wo,
                    ci_g,
                    hi,
                    wi,
                    kh,
                    kw,
                    kh0,
                    kw0,
                    liveh,
                    livew,
                    sh,
                    sw,
                    resh,
                    resw,
                    ph,
                    pw,
                    dh,
                    dw,
                    upcast,
                    bn,
                    bci,
                    bco,
                    num_warps=warps,
                    num_stages=stages,
                )
    return grad_input


def _wgrad_tiles(dtype, co, wo, nh_total):
    """(BLOCK_M, BLOCK_J, BLOCK_W, NHPC, num_warps, num_stages) for wgrad.

    GCU400 sweep winners: one flat tile shape took every benchmark cell -- a
    32-lane output-channel tile, a 128-wide (ci, kh, kw) column tile, 64 wo
    lanes per dot, 2 warps and a single stage.  Small (n, ho) planes keep 8
    unrolled rows per program so the split axis still fills the device.
    """
    return 32, 128, 64, 16, 2, 1


def _wgrad_fast(
    grad_output, input, grad_weight, grad_bias, groups, sh, sw, ph, pw, dh, dw
):
    """grad_weight (+ grad_bias piggybacked on the same split-K slabs)."""
    n, ci_total, h, w = input.shape
    ci_g = ci_total // groups
    co_g = grad_output.shape[1] // groups
    kh, kw = grad_weight.shape[2], grad_weight.shape[3]
    ho, wo = grad_output.shape[2], grad_output.shape[3]
    has_b = grad_bias is not None
    kk = kh * kw
    cikk = ci_g * kk
    sgo_n = grad_output.shape[1] * ho * wo
    sx_n = ci_total * h * w
    nh_total = n * ho

    bm, bj, bw, nhpc, warps, stages = _wgrad_tiles(input.dtype, co_g, wo, nh_total)
    w_chunks = triton.cdiv(wo, bw)
    m_tiles = triton.cdiv(co_g, bm)
    j_tiles = triton.cdiv(cikk, bj)
    while nhpc * w_chunks > 64:
        nhpc //= 2
    split_k = triton.cdiv(nh_total, nhpc)
    part = torch.empty(
        (groups, split_k, co_g * cikk),
        dtype=torch.float32,
        device=input.device,
    )
    bpart = (
        torch.empty((groups, split_k, co_g), dtype=torch.float32, device=input.device)
        if has_b
        else part
    )
    upcast = grad_output.dtype != torch.float32
    for g in range(groups):
        goff = g * co_g * ho * wo
        xoff = g * ci_g * h * w
        poff = g * split_k * co_g * cikk
        bpoff = g * split_k * co_g
        grid = (split_k, j_tiles, m_tiles)
        _cbo_wgrad_dot[grid](
            grad_output,
            input,
            part,
            bpart,
            goff,
            xoff,
            poff,
            bpoff,
            sgo_n,
            sx_n,
            nh_total,
            co_g,
            ci_g,
            n,
            ho,
            wo,
            h,
            w,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            has_b,
            upcast,
            nhpc,
            w_chunks,
            bm,
            bj,
            bw,
            num_warps=warps,
            num_stages=stages,
        )
        gwoff = g * co_g * cikk
        gboff = g * co_g
        reduce_grid = (triton.cdiv(co_g * cikk, 256),)
        _cbo_wgrad_reduce[reduce_grid](
            part,
            bpart,
            grad_weight,
            grad_bias if has_b else bpart,
            poff,
            bpoff,
            gwoff,
            gboff,
            co_g,
            ci_g,
            kh,
            kw,
            split_k,
            has_b,
            256,
            max(16, min(128, _next_pow2(split_k))),
            max(16, min(128, _next_pow2(co_g))),
            num_warps=4,
        )
    return grad_weight, grad_bias


# ---------------------------------------------------------------------------
# fallback path (original composable implementation, kept verbatim)
# ---------------------------------------------------------------------------
def _compute_bias_grad(out_grad, dtype):
    # out_grad: [N, C, H, W]. grad_bias[c] = sum over n, h, w of out_grad[n, c, h, w].
    # Routed through the FlagGems sum_dim kernel with an fp32 accumulator,
    # matching the reduction the gems conv2d backward uses for its own bias
    # gradient (host-function convention: no native PyTorch compute).
    return gems_sum_dim(out_grad, dim=[0, 2, 3], dtype=torch.float32).to(dtype)


def _conv2d_grad_input(
    out_grad, weight, input_shape, stride, padding, dilation, groups
):
    # grad_input is exactly the transposed convolution of out_grad by the same
    # weight: d/d input of conv2d(input, weight) is conv_transpose2d(grad, weight).
    # conv_transpose2d is a first-class FlagGems operator (a Triton kernel), so the
    # whole path stays inside gems instead of falling back to a PyTorch conv.
    #
    # A plain stride-s transpose enlarges the output to
    #   (Ho - 1) * s - 2p + d * (k - 1) + 1,
    # which can fall short of the true input size when stride > 1 truncated the
    # forward output; output_padding makes up exactly that difference so the
    # returned gradient matches input_shape element-for-element.
    _, _, weight_height, weight_width = weight.shape
    _, _, input_height, input_width = input_shape

    stride_height, stride_width = stride
    padding_height, padding_width = padding
    dilation_height, dilation_width = dilation

    out_height, out_width = out_grad.shape[2], out_grad.shape[3]
    base_height = (
        (out_height - 1) * stride_height
        - 2 * padding_height
        + dilation_height * (weight_height - 1)
        + 1
    )
    base_width = (
        (out_width - 1) * stride_width
        - 2 * padding_width
        + dilation_width * (weight_width - 1)
        + 1
    )
    output_padding = (input_height - base_height, input_width - base_width)

    grad_input = conv_transpose2d(
        out_grad,
        weight,
        None,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
    )
    return grad_input


def _conv2d_grad_weight(
    out_grad, input, weight_shape, stride, padding, dilation, groups
):
    # grad_weight[co, ci, kh, kw] = sum over (n, ho, wo) of
    #   out_grad[n, co, ho, wo] * input_patch[n, ci, kh, kw, ho, wo].
    #
    # Reuse the gems conv2d weight-gradient Triton kernel directly. It streams the
    # input once and contracts against out_grad with an fp32 accumulator, so the
    # result clears the fp32 backward tolerance without materializing im2col columns
    # and without touching any PyTorch conv primitive.
    out_c, weight_c, weight_height, weight_width = weight_shape
    # out_grad channels carry the full (groups * out_c) count; the kernel's out_c
    # constexpr is the per-group output-channel count.
    out_c_per_group = out_grad.shape[1] // groups

    in_n, _, input_height, input_width = input.shape
    out_height, out_width = out_grad.shape[2], out_grad.shape[3]

    stride_height, stride_width = stride
    padding_height, padding_width = padding
    dilation_height, dilation_width = dilation

    input = input.contiguous()
    out_grad = out_grad.contiguous()

    grad_weight = torch.zeros(
        out_c,
        weight_c,
        weight_height,
        weight_width,
        dtype=torch.float32,
        device=input.device,
    )

    grid_weight = lambda meta: (
        triton.cdiv(weight_c * weight_height * weight_width, meta["BLOCK_CI_HK_WK"]),
        groups,
        triton.cdiv(out_c_per_group, meta["BLOCK_CO"]),
    )
    conv2d_backward_kernel_weight[grid_weight](
        input,
        out_grad,
        grad_weight,
        *input.stride(),
        *grad_weight.stride(),
        *out_grad.stride(),
        input_height,
        input_width,
        weight_height,
        weight_width,
        weight_c,
        in_n,
        stride_height,
        stride_width,
        out_height,
        out_width,
        out_c_per_group,
        padding_height,
        padding_width,
        dilation_height,
        dilation_width,
    )
    return grad_weight


def _normalize_pair(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return (int(value[0]), int(value[0]))
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


def _backward_core(
    grad_output,
    input,
    weight,
    stride,
    padding,
    dilation,
    groups,
    output_mask,
    out0,
    out1,
    out2,
):
    stride_height, stride_width = stride
    padding_height, padding_width = padding
    dilation_height, dilation_width = dilation
    co_g = weight.shape[0] // groups
    ci_g = weight.shape[1]

    dtype = input.dtype
    fast = (
        dtype in _FAST_DTYPES
        and grad_output.dtype == dtype
        and weight.dtype == dtype
        and grad_output.is_contiguous()
        and input.is_contiguous()
        and weight.is_contiguous()
        and _int32_safe(grad_output, input)
        and weight.numel() * 96 * 4 < 2**31
    )
    if out0 is not None:
        fast = fast and out0.dtype == dtype and out0.is_contiguous()
    if out1 is not None:
        fast = fast and out1.dtype == dtype and out1.is_contiguous()
    if out2 is not None:
        fast = fast and out2.dtype == dtype and out2.is_contiguous()
    # The residue dgrad kernel needs enough lanes for its dot tiles.
    dgrad_fast = (
        fast and co_g >= 16 and ci_g >= 16 and input.shape[2] * input.shape[3] >= 16
    )

    grad_input = None
    grad_weight = None
    grad_bias = None

    tops = _try_topsaten_conv_backward(
        grad_output,
        input,
        weight,
        (stride_height, stride_width),
        (padding_height, padding_width),
        (dilation_height, dilation_width),
        groups,
        output_mask,
        out0,
        out1,
        out2,
    )
    if tops is not None:
        return tops

    if output_mask[0]:
        if out0 is not None:
            grad_input = out0
        else:
            grad_input = torch.empty_like(input)
        if dgrad_fast:
            _dgrad_fast(
                grad_output,
                weight,
                grad_input,
                groups,
                stride_height,
                stride_width,
                padding_height,
                padding_width,
                dilation_height,
                dilation_width,
            )
        else:
            grad_input = _conv2d_grad_input(
                grad_output,
                weight,
                input.shape,
                stride,
                padding,
                dilation,
                groups,
            ).to(dtype)
            if out0 is not None:
                out0.copy_(grad_input)
                grad_input = out0

    if output_mask[1] or output_mask[2]:
        if fast and output_mask[1] and output_mask[2]:
            if out1 is not None:
                grad_weight = out1
            else:
                grad_weight = torch.empty_like(weight)
            if out2 is not None:
                grad_bias = out2
            else:
                grad_bias = torch.empty(
                    weight.shape[0], dtype=dtype, device=input.device
                )
            _wgrad_fast(
                grad_output,
                input,
                grad_weight,
                grad_bias,
                groups,
                stride_height,
                stride_width,
                padding_height,
                padding_width,
                dilation_height,
                dilation_width,
            )
        else:
            if output_mask[1]:
                grad_weight = _conv2d_grad_weight(
                    grad_output, input, weight.shape, stride, padding, dilation, groups
                ).to(dtype)
                if out1 is not None:
                    out1.copy_(grad_weight)
                    grad_weight = out1
            if output_mask[2]:
                grad_bias = _compute_bias_grad(grad_output, dtype)
                if out2 is not None:
                    out2.copy_(grad_bias)
                    grad_bias = out2

    return grad_input, grad_weight, grad_bias


def convolution_backward_overrideable(
    grad_output,
    input,
    weight,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    logger.debug("GEMS_ENFLAME CONVOLUTION_BACKWARD_OVERRIDEABLE")
    if transposed:
        raise RuntimeError(
            "convolution_backward_overrideable: transposed not supported"
        )
    if input.ndim != 4 or weight.ndim != 4:
        raise RuntimeError(
            "convolution_backward_overrideable currently supports 2D convolution only"
        )

    stride = _normalize_pair(stride)
    padding = _normalize_pair(padding)
    dilation = _normalize_pair(dilation)
    groups = int(groups)

    return _backward_core(
        grad_output,
        input,
        weight,
        stride,
        padding,
        dilation,
        groups,
        output_mask,
        None,
        None,
        None,
    )


def convolution_backward_overrideable_out(
    grad_output,
    input,
    weight,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
    out0,
    out1,
    out2,
):
    logger.debug("GEMS_ENFLAME CONVOLUTION_BACKWARD_OVERRIDEABLE_OUT")
    if transposed:
        raise RuntimeError(
            "convolution_backward_overrideable: transposed not supported"
        )
    if input.ndim != 4 or weight.ndim != 4:
        raise RuntimeError(
            "convolution_backward_overrideable currently supports 2D convolution only"
        )

    stride = _normalize_pair(stride)
    padding = _normalize_pair(padding)
    dilation = _normalize_pair(dilation)
    groups = int(groups)

    _backward_core(
        grad_output,
        input,
        weight,
        stride,
        padding,
        dilation,
        groups,
        output_mask,
        out0,
        out1,
        out2,
    )
    return out0, out1, out2
