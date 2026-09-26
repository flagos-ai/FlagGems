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

from .conv2d import conv2d

logger = logging.getLogger(__name__)

_ZERO_BIAS_CACHE = {}


def _zero_bias(out_c, device):
    key = (out_c, str(device))
    t = _ZERO_BIAS_CACHE.get(key)
    if t is None:
        t = torch.zeros(out_c, device=device, dtype=torch.float)
        _ZERO_BIAS_CACHE[key] = t
    return t


_ENABLED = os.environ.get("FG_DEPTHWISE_FASTPATH", "1") != "0"
_LOCK = threading.Lock()
_CACHE = {}
_BLOCKED = set()
_MAX_KEYS = 64


def _launcher_cls():
    try:
        drv_cfg = getattr(importlib.import_module("triton.runtime"), "driver")
        return getattr(drv_cfg.active, "launcher_cls_xpu", None) or getattr(
            drv_cfg.active, "launcher_cls", None
        )
    except Exception:
        return None


def _base_eligible(input, weight):
    if not _ENABLED:
        return False
    if not (input.is_cuda and weight.is_cuda):
        return False
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if input.dtype != weight.dtype:
        return False
    if input.requires_grad or weight.requires_grad:
        return False
    return True


def _key_of(input, weight, device, stride, padding, dilation):
    return (
        tuple(input.shape),
        tuple(input.stride()),
        input.dtype,
        tuple(weight.shape),
        tuple(weight.stride()),
        device.index,
        str(stride),
        str(padding),
        str(dilation),
    )


def _capture_and_delegate(plain):
    """Run the plain path once, capturing the deepest launcher arguments."""
    launcher_cls = _launcher_cls()
    if launcher_cls is None:
        return plain(), None
    cap = {}
    cap["orig_calls"] = launcher_cls.__call__

    def call_hook(lself, *a, **k):
        if "launcher" not in cap:
            cap["launcher"] = lself
            orig_launch = lself.launch

            def launch_hook(*la):
                if "a2" not in cap:
                    cap["a2"] = tuple(la)
                return orig_launch(*la)

            cap["orig_launch"] = orig_launch
            lself.launch = launch_hook
        return cap["orig_calls"](lself, *a, **k)

    launcher_cls.__call__ = call_hook
    try:
        out = plain()
    finally:
        launcher_cls.__call__ = cap["orig_calls"]
        lself = cap.get("launcher")
        if lself is not None and "orig_launch" in cap:
            try:
                lself.launch = cap["orig_launch"]
            except Exception:
                pass
    if "a2" not in cap:
        return out, None
    return out, cap


def _record(out, cap, input, weight):
    """Validate the captured args and build the replay record; None on doubt."""
    a2 = cap.get("a2")
    launcher = cap.get("launcher")
    if a2 is None or launcher is None:
        return None
    fn_name = str(getattr(launcher, "_fn_name", ""))
    if fn_name and "conv2d_forward" not in fn_name:
        return None
    tpos = [i for i, v in enumerate(a2) if isinstance(v, torch.Tensor)]
    if len(tpos) < 3:
        return None
    j_in, j_w, j_out = tpos[0], tpos[1], tpos[2]
    is_cast = input.dtype == torch.bfloat16 and a2[j_in].dtype == torch.float32
    if not is_cast:
        if a2[j_in].dtype != input.dtype:
            return None
        if tuple(a2[j_in].shape) != tuple(input.shape):
            return None
    if tuple(a2[j_w].shape) != tuple(weight.shape):
        return None
    if tuple(a2[j_out].shape) != tuple(out.shape):
        return None
    jb = tpos[3] if len(tpos) > 3 else None
    a2 = list(a2)
    a2[j_in] = None
    a2[j_w] = None
    a2[j_out] = None
    if jb is not None:
        a2[jb] = None
    return {
        "launcher": launcher,
        "a2": tuple(a2),
        "j_in": j_in,
        "j_w": j_w,
        "j_out": j_out,
        "jb": jb,
        "cast": is_cast,
        "out_shape": tuple(out.shape),
        "out_dtype": input.dtype,
        "ref": out.detach().clone(),
        "stream": torch.cuda.current_stream().cuda_stream,
        "verified": False,
    }


def _replay(rec, input, weight, bias):
    if torch.cuda.current_stream().cuda_stream != rec["stream"]:
        raise RuntimeError("stream changed")
    a = list(rec["a2"])
    if rec["cast"]:
        a[rec["j_in"]] = input.to(torch.float32)
        a[rec["j_w"]] = weight.to(torch.float32)
        o32 = torch.empty(rec["out_shape"], device=input.device, dtype=torch.float32)
        a[rec["j_out"]] = o32
        if rec["jb"] is not None:
            a[rec["jb"]] = bias
        rec["launcher"].launch(*a)
        return o32.to(rec["out_dtype"])
    o = torch.empty(rec["out_shape"], device=input.device, dtype=rec["out_dtype"])
    a[rec["j_in"]] = input
    a[rec["j_w"]] = weight
    a[rec["j_out"]] = o
    if rec["jb"] is not None:
        a[rec["jb"]] = bias
    rec["launcher"].launch(*a)
    return o


def _verify(rec, out):
    ref = rec["ref"]
    if ref is None:
        return True
    tol = {
        torch.float32: 1e-4,
        torch.float16: 1e-2,
        torch.bfloat16: 2e-2,
    }[rec["out_dtype"]]
    try:
        d = (out.detach().float() - ref.float()).abs().max().item()
    except Exception:
        return False
    if d <= tol:
        rec["verified"] = True
        rec["ref"] = None
        return True
    return False


def _conv_depthwise2d(input, weight, kernel_size, bias, stride, padding, dilation):
    logger.debug("GEMS_KUNLUNXIN CONV_DEPTHWISE2D")
    assert (
        input.ndim == 4
    ), "Invalid input tensor must be 4D, recevied shape {input.shape}"
    assert (
        weight.shape[0] % input.shape[1] == 0
    ), "Output channels must be multiple of input, recevied output {weught.shape[0], input {input.shape[0]}}"
    assert (
        weight.shape[1] == 1
    ), "input channels of per goups must be 1, recevied {weight.shape[1]}"
    groups = input.shape[1]
    user_bias = bias
    if bias is None:
        bias = _zero_bias(weight.shape[0], input.device)

    def plain():
        return conv2d(input, weight, bias, stride, padding, dilation, groups)

    if user_bias is not None or not _base_eligible(input, weight):
        return plain()

    device = input.device
    key = _key_of(input, weight, device, stride, padding, dilation)
    rec = _CACHE.get(key)
    if rec is not None:
        ok = False
        try:
            out = _replay(rec, input, weight, bias)
            ok = rec["verified"] or _verify(rec, out)
        except Exception:
            ok = False
        if ok:
            return out
        with _LOCK:
            _CACHE.pop(key, None)
            if len(_BLOCKED) < 256:
                _BLOCKED.add(key)
        return plain()

    if key in _BLOCKED:
        return plain()

    with _LOCK:
        if len(_CACHE) >= _MAX_KEYS:
            return plain()
        out, cap = _capture_and_delegate(plain)
        rec2 = _record(out, cap, input, weight) if cap else None
        if rec2 is None:
            if len(_BLOCKED) < 256:
                _BLOCKED.add(key)
            return out
        _CACHE[key] = rec2
    return out
