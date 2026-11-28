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
#
# Ascend implementation imported from the tuned operator batch
# host entry points below mirror the generic
# flag_gems names so the backend registrar can pick them up.

"""严格Triton舍入：继承KG第二轮已测数值和流式核，按special_round_out合同接收必需out并核对输出大小/设备。
不改设备公式；精度仍由冻结KGS合同及NPU补测判断。
"""

import torch
import torch_npu
import triton
import triton.language as tl

_MAGIC = 12582912.0  # 1.5 * 2**23
_CORES = {}
_STREAM = {}


@triton.jit
def _value(x, DECIMALS: tl.constexpr):
    xf = x.to(tl.float32)
    if DECIMALS == 0:
        return (xf + 12582912.0) - 12582912.0
    scale: tl.constexpr = 10.0**DECIMALS
    v = xf * scale
    return tl.div_rn((v + 12582912.0) - 12582912.0, scale)


@triton.jit
def _round_stream(X, Y, N, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    full = (N // step) * step
    for start in tl.range(pid * BLOCK, full, step):
        x = tl.load(X + start + lane)
        tl.store(Y + start + lane, _value(x, DECIMALS).to(Y.dtype.element_ty))
    start = full + pid * BLOCK
    if start < N:
        idx = start + lane
        m = idx < N
        x = tl.load(X + idx, m, other=0.0)
        tl.store(Y + idx, _value(x, DECIMALS).to(Y.dtype.element_ty), m)


@triton.jit
def _round_strided(
    X,
    Y,
    N,
    SHAPE: tl.constexpr,
    XS: tl.constexpr,
    YS: tl.constexpr,
    DECIMALS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remainder = index
    xi = tl.full((BLOCK,), 0, tl.int64)
    yi = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coordinate = remainder % SHAPE[axis]
        remainder = remainder // SHAPE[axis]
        xi += coordinate.to(tl.int64) * XS[axis]
        yi += coordinate.to(tl.int64) * YS[axis]
    x = tl.load(X + xi, index < N, other=0.0)
    tl.store(Y + yi, _value(x, DECIMALS).to(Y.dtype.element_ty), index < N)


def _stream_getter(device):
    getter = _STREAM.get(device)
    if getter is not None:
        return getter
    raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    if raw is not None:
        try:
            expected = int(torch.npu.current_stream(device).npu_stream)
            try:
                if int(raw(device)) == expected:
                    getter = lambda: raw(device)
            except TypeError:
                if int(raw()) == expected:
                    getter = raw
        except Exception:
            getter = None
    if getter is None:
        from triton.runtime import driver

        getter = lambda: driver.active.get_current_stream(device)
    _STREAM[device] = getter
    return getter


def _cores(device):
    cores = _CORES.get(device)
    if cores is None:
        cores = 1
        try:
            from triton.runtime import driver

            props = driver.active.utils.get_device_properties(device)
            cores = max(
                1, int(props.get("num_vectorcore") or props.get("num_aicore") or 1)
            )
        except Exception:
            pass
        _CORES[device] = cores
    return cores


def _geometry(n, device):
    if n > 8192:
        block = 8192
    else:
        block = max(64, 1 << (n - 1).bit_length())
    grid = min(_cores(device), (n + block - 1) // block)
    return block, max(1, grid)


def _build(x, out, n, decimals, device):
    block, grid = _geometry(n, device)
    launch = fn = packed = getter = None
    try:
        compiled = _round_stream.warmup(x, out, n, decimals, block, grid=(1,))
        runner = compiled.run
        md = runner.metadata
        if not (
            getattr(runner, "compile_only", False)
            or getattr(runner, "enable_msprof_register_tensor", False)
            or getattr(md, "debug_enabled", False)
            or getattr(md, "debug", False)
        ):
            launch = runner.launch
            fn = compiled.function
            packed = compiled.packed_metadata
            getter = _stream_getter(device)
    except Exception:
        pass
    return [
        x.dtype,
        out.dtype,
        n,
        device,
        decimals,
        block,
        grid,
        launch,
        fn,
        packed,
        getter,
    ]


# 0 in_dtype, 1 out_dtype, 2 n, 3 device, 4 decimals, 5 block, 6 grid,
# 7 launch, 8 function, 9 packed, 10 stream getter
_FAST = [None, None, -1, -1, 0, 0, 1, None, None, None, None]
_PLANS = {}


def _slow(x, decimals, out):
    n = x.numel()
    if not n:
        return out
    if not x.is_contiguous() or not out.is_contiguous():
        _round_strided[(triton.cdiv(n, 256),)](
            x,
            out,
            n,
            tuple(x.shape),
            tuple(x.stride()),
            tuple(out.stride()),
            decimals,
            BLOCK=256,
        )
        return out
    device = x.get_device()
    key = (device, x.dtype, out.dtype, n, decimals)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _PLANS[key] = _build(x, out, n, decimals, device)
    _FAST[:] = plan
    if plan[7] is not None:
        xp = x.data_ptr()
        yp = out.data_ptr()
        if not ((xp | yp) & 15):
            plan[7](
                plan[6],
                1,
                1,
                plan[10](),
                plan[8],
                plan[9],
                None,
                None,
                None,
                xp,
                yp,
                n,
                decimals,
                plan[5],
            )
            return out
    _round_stream[plan[6],](x, out, n, decimals, BLOCK=plan[5])
    return out


def run(self, out, *, decimals=0):
    input = self
    if out is None:
        out = torch.empty_like(input, memory_format=torch.contiguous_format)
    if out.shape != input.shape:
        out.resize_(input.shape)
    if out.get_device() != input.get_device():
        raise ValueError("输入与out设备不同")
    f = _FAST
    if (
        input.dtype is f[0]
        and f[7] is not None
        and decimals == f[4]
        and out is not None
        and out.dtype is f[1]
        and input.numel() == f[2]
        and input.get_device() == f[3]
        and input.is_contiguous()
        and out.is_contiguous()
    ):
        xp = input.data_ptr()
        yp = out.data_ptr()
        if not ((xp | yp) & 15):
            f[7](
                f[6],
                1,
                1,
                f[10](),
                f[8],
                f[9],
                None,
                None,
                None,
                xp,
                yp,
                f[2],
                decimals,
                f[5],
            )
            return out
    return _slow(input, decimals, out)


# --------------------------------------------------------------------------- #
# FlagGems entry points
# --------------------------------------------------------------------------- #
def special_round_out(*args, **kwargs):
    return run(*args, **kwargs)
