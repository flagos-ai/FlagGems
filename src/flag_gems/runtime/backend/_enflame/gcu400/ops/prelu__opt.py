"""prelu 候选 v2：大 BLOCK 分档 + 位移替代运行时整数除法。

生产 gcu400 `prelu_kernel` 有两个瓶颈：
1. BLOCK 分档只按 `n/1024 > 65535` 二选一（1024 或 32768），
   2^24 级 shape 仍只有 1024 的 tile，并行度不足；
2. 逐元素 `(offsets // S) % C` 的 S/C 是运行时参数，无法编译期折叠，
   每个元素都要做完整整数除法。

benchmark 的 S/C 均为 2 的幂，可用 `(off >> log2(S)) & (C-1)` 等价替代。
S/C 保持运行时参数，只把 USE_SHIFT/LOG2_S/C_MASK 提为 constexpr —— 已验证
直接把 S/C 本身设为 constexpr 在 BLOCK>=32768 时会被 GCU 后端编译出错。
非 2 的幂时自动回退运行时除法，保证通用性。

小 shape 沿用生产的 BLOCK=1024/nw4，避免微秒级 case 回归；
大 shape 用 B131072/nw4（B262144 会触发 dsm memory OutOfResources，排除）。
"""
import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)


@triton.jit
def _prelu_kernel(
    x_ptr, w_ptr, out_ptr,
    n_elements, S, C,
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
    result = tl.where(x >= 0, x, alpha * x)
    tl.store(out_ptr + out_offset, result, mask=in_range)


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def prelu(*args, **kwargs):
    logger.debug("GEMS_ENFLAME PRELU (shift + block tiling)")
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

    use_shift = _is_power_of_two(spatial) and _is_power_of_two(channels)
    log2_spatial = spatial.bit_length() - 1 if use_shift else 0
    channel_mask = channels - 1 if use_shift else 0

    if n_elements < 65536:
        BLOCK_SIZE, num_warps = 1024, 4
    else:
        BLOCK_SIZE, num_warps = 131072, 4
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    if grid[0] > 65535:
        BLOCK_SIZE = 32768
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _prelu_kernel[grid](
        x, weight, out, n_elements, spatial, channels,
        int(use_shift), log2_spatial, channel_mask, int(w_is_scalar),
        BLOCK_SIZE, num_warps=num_warps,
    )
    return out
