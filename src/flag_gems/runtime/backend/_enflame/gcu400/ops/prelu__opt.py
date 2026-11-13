"""prelu 候选 v3：按 weight/channel 结构四路分派，消除逐元素 weight gather。

生产 v2 的唯一热路径是逐元素 ``tl.load(w_ptr + channel)``：当块内 channel
地址恒定或跨步重复（S=512/1024 的 core 形状）时，GCU 后端把 gather 标量化，
(64,512,512) 只有 0.13x、(1024,1024,1024) 只有 0.049x（do_bench 口径，
去掉 w-load 的消融地板分别是 27.3us / 1389.5us）。

v3 分派（n >= 65536 才进新路径，小形状沿用 gather 保持 1.39x）：
1. scalar weight：平铺免 mask 全块 + 单尾块（v1s，b131072/w4 → 1.00-1.04x）；
2. S >= 2 per-channel：行块 tile [BLOCK_R, BLOCK_S]，alpha = w[rows % C]
   一次 [BLOCK_R] gather + 行广播（fp16/bf16 tile 131072/nw2、fp32 tile
   65536/nw2 → (64,512,512) 2.52x/2.19x/1.70x，1024^3 1.02x/0.84x/0.81x）；
3. S == 1 per-channel：通道 tile [BLOCK_R, BLOCK_C]，alpha = w[cols] 连续
   向量 + 列广播（(4096,4096) 0.87x/0.81x——2D 寻址税地板 37.3us）；
4. 其余（n < 65536、grid 超限）：v2 gather 路径原样兜底（pow2 走位移）。

已否决（本轮实测）：循环平铺 v5 0.22-0.33x（深循环小 tile 无流水）、
reshape 广播 v6 0.61x + fp32 DSM 三倍物化 OOR、每行平铺 v9 0.15x
（w 向量逐 program 重载不命中缓存）。alpha 行广播物化第 3 个 DSM tile，
故 tile 上限 131072 (fp16/bf16) / 65536 (fp32)，防 OutOfResources。
"""

import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)


@triton.jit
def _prelu_scalar_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_elements,
    n_full,
    BLOCK_SIZE: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    alpha = tl.load(w_ptr)
    if HAS_TAIL:
        if pid * BLOCK_SIZE < n_full:
            x = tl.load(x_ptr + offsets)
            tl.store(out_ptr + offsets, tl.where(x >= 0, x, alpha * x))
        else:
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=0)
            tl.store(out_ptr + offsets, tl.where(x >= 0, x, alpha * x), mask=mask)
    else:
        x = tl.load(x_ptr + offsets)
        tl.store(out_ptr + offsets, tl.where(x >= 0, x, alpha * x))


@triton.jit
def _prelu_rowtile_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    R,
    C,
    S,
    BLOCK_R: tl.constexpr,
    BLOCK_S: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_s = tl.program_id(1)
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    alpha = tl.load(w_ptr + rows % C)
    ptrs = rows[:, None] * S + cols[None, :]
    if EVEN:
        x = tl.load(x_ptr + ptrs)
        tl.store(out_ptr + ptrs, tl.where(x >= 0, x, alpha[:, None] * x))
    else:
        mask = (rows < R)[:, None] & (cols < S)[None, :]
        x = tl.load(x_ptr + ptrs, mask=mask, other=0)
        tl.store(out_ptr + ptrs, tl.where(x >= 0, x, alpha[:, None] * x), mask=mask)


@triton.jit
def _prelu_channel_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    R,
    C,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    alpha = tl.load(w_ptr + cols, mask=cols < C, other=0)
    ptrs = rows[:, None] * C + cols[None, :]
    if EVEN:
        x = tl.load(x_ptr + ptrs)
        tl.store(out_ptr + ptrs, tl.where(x >= 0, x, alpha[None, :] * x))
    else:
        mask = (rows < R)[:, None] & (cols < C)[None, :]
        x = tl.load(x_ptr + ptrs, mask=mask, other=0)
        tl.store(out_ptr + ptrs, tl.where(x >= 0, x, alpha[None, :] * x), mask=mask)


@triton.jit
def _prelu_gather_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_elements,
    S,
    C,
    USE_SHIFT: tl.constexpr,
    LOG2_S: tl.constexpr,
    C_MASK: tl.constexpr,
    W_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    out_offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_range = out_offset < n_elements
    x = tl.load(x_ptr + out_offset, mask=in_range, other=0)
    if W_SCALAR:
        alpha = tl.load(w_ptr)
    else:
        if USE_SHIFT:
            channel = (out_offset >> LOG2_S) & C_MASK
        else:
            channel = (out_offset // S) % C
        alpha = tl.load(w_ptr + channel, mask=in_range, other=0)
    tl.store(out_ptr + out_offset, tl.where(x >= 0, x, alpha * x), mask=in_range)


def _pow2_floor(value):
    p = 1
    while p * 2 <= value:
        p *= 2
    return p


def _next_pow2(value):
    p = 1
    while p < value:
        p *= 2
    return p


def _launch_gather(x, weight, out, n_elements, S, C, w_is_scalar):
    block = 131072 if n_elements >= 65536 else 1024
    grid = (triton.cdiv(n_elements, block),)
    if grid[0] > 65535:
        block = 32768
        grid = (triton.cdiv(n_elements, block),)
    use_shift = S > 0 and (S & (S - 1)) == 0 and C > 0 and (C & (C - 1)) == 0
    _prelu_gather_kernel[grid](
        x,
        weight,
        out,
        n_elements,
        S,
        C,
        int(use_shift),
        S.bit_length() - 1 if use_shift else 0,
        C - 1 if use_shift else 0,
        int(w_is_scalar),
        block,
        num_warps=4,
    )


def prelu(*args, **kwargs):
    logger.debug("GEMS_ENFLAME PRELU (v3 structural dispatch)")
    if len(args) >= 2:
        x, weight = args[0], args[1]
    else:
        x = kwargs.get("input", kwargs.get("self"))
        weight = kwargs.get("weight")
    if x is None or weight is None:
        raise ValueError("prelu expects (input, weight) as arguments.")
    if x.device.type != flag_gems.device or weight.device.type != flag_gems.device:
        raise AssertionError(f"Tensors must be {flag_gems.device} tensors.")
    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    x = x.contiguous()
    weight = weight.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out

    ndim = x.dim()
    if weight.numel() == 1:
        channels, spatial, w_is_scalar = 1, 1, True
    else:
        if ndim == 0:
            raise AssertionError("Non-scalar weight provided for a 0-dim input.")
        if ndim == 1:
            channels, spatial = x.shape[0], 1
        else:
            channels, spatial = x.shape[1], 1
            if ndim > 2:
                for dim in x.shape[2:]:
                    spatial *= dim
        if weight.numel() != channels:
            raise AssertionError(
                f"Weight numel ({weight.numel()}) must equal channel dimension size ({channels})."
            )
        w_is_scalar = False
    channels = max(int(channels), 1)
    spatial = max(int(spatial), 1)

    if n_elements < 65536:
        _launch_gather(x, weight, out, n_elements, spatial, channels, w_is_scalar)
        return out

    if w_is_scalar:
        block = 131072
        n_full = n_elements // block * block
        has_tail = 1 if n_elements % block else 0
        grid = (n_elements // block + has_tail,)
        _prelu_scalar_kernel[grid](
            x,
            weight,
            out,
            n_elements,
            n_full,
            BLOCK_SIZE=block,
            HAS_TAIL=has_tail,
            num_warps=4,
        )
        return out

    elem_size = x.element_size()
    if elem_size < 4:
        tile, max_s_block = 131072, 512
    else:
        tile, max_s_block = 65536, 1024

    if spatial >= 2:
        block_s = min(_pow2_floor(spatial), max_s_block)
        block_r = max(1, min(tile // block_s, 512))
        rows = n_elements // spatial
        grid = (triton.cdiv(rows, block_r), triton.cdiv(spatial, block_s))
        if grid[0] <= 65535 and grid[1] <= 65535:
            even = rows % block_r == 0 and spatial % block_s == 0
            _prelu_rowtile_kernel[grid](
                x,
                weight,
                out,
                rows,
                channels,
                spatial,
                BLOCK_R=block_r,
                BLOCK_S=block_s,
                EVEN=int(even),
                num_warps=2,
            )
            return out
    else:
        rows = n_elements // channels
        block_c = min(_next_pow2(channels), 4096)
        block_r = max(1, min(tile // block_c, _next_pow2(rows)))
        grid = (triton.cdiv(rows, block_r), triton.cdiv(channels, block_c))
        if grid[0] <= 65535 and grid[1] <= 65535:
            even = rows % block_r == 0 and channels % block_c == 0
            _prelu_channel_kernel[grid](
                x,
                weight,
                out,
                rows,
                channels,
                BLOCK_R=block_r,
                BLOCK_C=block_c,
                EVEN=int(even),
                num_warps=4,
            )
            return out

    _launch_gather(x, weight, out, n_elements, spatial, channels, w_is_scalar)
    return out
