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

import importlib
import logging
import os
import threading

import torch

from .conv1d import conv1d
from .conv2d import conv2d
from .conv3d import conv3d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# [0920 C-190] aten::cudnn_convolution host fast path (capture/replay, same
# pattern as the conv_depthwise2d overlay).  Small-shape calls are dominated
# by host-side preparation (schema handling, launch assembly); on the first
# call for a config we delegate to the plain path and capture the deepest
# launcher arguments; later calls replay them with fresh tensor arguments
# spliced back in.  Any doubt (grad tensors, non-device tensors, string
# padding, non-square inputs, ambiguous captures) falls back to the plain
# path.  FG_CUDNNCONV_FASTPATH=0 disables.
# ---------------------------------------------------------------------------
_ENABLED = os.environ.get("FG_CUDNNCONV_FASTPATH", "1") != "0"
_LOCK = threading.Lock()
_CACHE = {}
_MAX_KEYS = 64
_LAST = None
_FAILS = {}


def _fp_norm(p):
    if isinstance(p, (list, tuple)):
        return tuple(p)
    return (p, p)


def _fp_eligible(input, weight, padding, groups):
    try:
        return (
            not isinstance(padding, str)
            and input.shape[-2] == input.shape[-1]
            and input.is_cuda
            and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and weight.dtype == input.dtype
            and not input.requires_grad
            and not weight.requires_grad
            and isinstance(groups, int)
        )
    except Exception:
        return False


def _fp_key(input, weight, padding, stride, dilation, groups):
    return (
        tuple(input.shape),
        tuple(input.stride()),
        input.dtype,
        tuple(weight.shape),
        tuple(weight.stride()),
        _fp_norm(stride),
        _fp_norm(padding),
        _fp_norm(dilation),
        int(groups),
    )


def _fp_match(a, tensors):
    out = []
    for t in tensors:
        hit = None
        for i, x in enumerate(a):
            if isinstance(x, torch.Tensor) and x.data_ptr() == t.data_ptr():
                if hit is not None:
                    hit = None  # ambiguous
                    break
                hit = i
        if hit is None:
            return None
        out.append(hit)
    return out


def _fp_capture(call):
    cap = {}
    drv_cfg = getattr(importlib.import_module("triton.runtime"), "driver")
    launcher_cls = getattr(drv_cfg.active, "launcher_cls_xpu", None) or getattr(
        drv_cfg.active, "launcher_cls", None
    )
    orig_calls = launcher_cls.__call__ if launcher_cls is not None else None
    launches = []

    def call_hook(launcher_self, *a, **k):
        try:
            orig_launch = launcher_self.launch

            def launch_hook(*la):
                launches.append(1)
                if "a2" not in cap:
                    cap["a2"] = tuple(la)
                    cap["orig_launch"] = orig_launch
                return orig_launch(*la)

            launcher_self.launch = launch_hook
            try:
                return orig_calls(launcher_self, *a, **k)
            finally:
                launcher_self.launch = orig_launch
        except Exception:
            return orig_calls(launcher_self, *a, **k)

    if orig_calls is not None:
        launcher_cls.__call__ = call_hook
    try:
        ret = call()
    finally:
        if orig_calls is not None:
            launcher_cls.__call__ = orig_calls
    cap["launches"] = len(launches)
    return ret, cap


def _fp_replay(rec, input, weight):
    a2 = list(rec["a2"])
    out = torch.empty(rec["out_shape"], dtype=rec["out_dtype"], device=input.device)
    a2[rec["j_x"]] = input
    a2[rec["j_w"]] = weight
    a2[rec["j_o"]] = out
    rec["orig_launch"](*a2)
    return out


def _spatial_tuple(value, dimensions, name):
    if isinstance(value, int):
        return (value,) * dimensions
    if isinstance(value, (list, tuple)) and len(value) == dimensions:
        return tuple(value)
    raise ValueError(f"{name} must have {dimensions} values, got {value}")


def _plain(input, weight, padding, stride, dilation, groups):
    dimensions = input.ndim - 2
    if dimensions not in (1, 2, 3):
        raise ValueError(
            f"cudnn_convolution expects a 3D, 4D, or 5D input, got {input.ndim}D"
        )

    padding = _spatial_tuple(padding, dimensions, "padding")
    stride = _spatial_tuple(stride, dimensions, "stride")
    dilation = _spatial_tuple(dilation, dimensions, "dilation")
    if dimensions == 1:
        return conv1d(input, weight, None, stride, padding, dilation, groups)
    if dimensions == 2:
        return conv2d(input, weight, None, stride, padding, dilation, groups)
    return conv3d(input, weight, None, stride, padding, dilation, groups)


def cudnn_convolution(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    """CUDNN-compatible no-bias convolution using native Kunlunxin kernels."""
    logger.debug("GEMS_KUNLUNXIN CUDNN_CONVOLUTION")
    if (
        input.ndim != 4
        or not _ENABLED
        or not _fp_eligible(input, weight, padding, groups)
    ):
        return _plain(input, weight, padding, stride, dilation, groups)

    global _LAST
    params = (_fp_norm(stride), _fp_norm(padding), _fp_norm(dilation), int(groups))
    last = _LAST
    rec = None
    kk = None
    if last is not None and input is last[0] and weight is last[1]:
        cand = last[2]
        if cand is not None and cand["params"] == params:
            rec = cand
    if rec is None:
        kk = _fp_key(input, weight, padding, stride, dilation, groups)
        with _LOCK:
            rec = _CACHE.get(kk)
        if rec is not None:
            _LAST = (input, weight, rec)
    if rec is not None:
        try:
            return _fp_replay(rec, input, weight)
        except Exception:
            with _LOCK:
                _CACHE.pop(kk, None)
            rec = None
    if kk is None:
        kk = _fp_key(input, weight, padding, stride, dilation, groups)
    if _FAILS.get(kk, 0) >= 2:
        return _plain(input, weight, padding, stride, dilation, groups)
    ret, cap = _fp_capture(
        lambda: _plain(input, weight, padding, stride, dilation, groups)
    )
    a2 = cap.get("a2")
    if (
        a2 is not None
        and cap.get("launches") == 1
        and torch.is_tensor(ret)
        and ret.is_contiguous()
    ):
        jdx = _fp_match(a2, [input, weight, ret])
        extra = (
            [
                x
                for i, x in enumerate(a2)
                if isinstance(x, torch.Tensor) and i not in (jdx[0], jdx[1], jdx[2])
            ]
            if jdx is not None
            else None
        )
        if jdx is not None and all(
            x.dtype == torch.float32 and x.numel() <= 4096 for x in extra
        ):
            rec = {
                "a2": a2,
                "orig_launch": cap["orig_launch"],
                "j_x": jdx[0],
                "j_w": jdx[1],
                "j_o": jdx[2],
                "out_shape": tuple(ret.shape),
                "out_dtype": ret.dtype,
                "params": params,
            }
            with _LOCK:
                if len(_CACHE) >= _MAX_KEYS:
                    _CACHE.clear()
                _CACHE[kk] = rec
                _FAILS.pop(kk, None)
            _LAST = (input, weight, rec)
        else:
            with _LOCK:
                _FAILS[kk] = _FAILS.get(kk, 0) + 1
    else:
        with _LOCK:
            _FAILS[kk] = _FAILS.get(kk, 0) + 1
    return ret
