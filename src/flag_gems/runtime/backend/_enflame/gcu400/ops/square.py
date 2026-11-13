"""square 候选 v3：flat grid-stride 家族制式重写（rad2deg/special_exp2 同款）。

v1/v2 用 one-program-per-chunk（grid=cdiv(n,BLOCK)，BLOCK 16384/32768，
nw 2/4）：512M fp16 实测 1.169ms = 1.84TB/s（native 0.719ms = 2.99TB/s），
官方 core geomean 0.867x。消融阶梯（do_bench kernel median，warmup 1000ms/
rep 100ms）证明 body 免费（copy 地板 680µs ≈ cast 680µs ≈ hp 682µs），
结构付费：同 body 换 48-SIP persistent grid-stride + BLOCK 65536/131072 +
nw2 后 512M fp16 676µs = 3.18TB/s 贴地板（1.06x native）、fp32 1348µs
（1.11x）、16.7M fp16 27.2µs（1.09x）。hp 体（fp32 计算后窄回）与公版
opmath 一致且 bit-exact（torch.equal 验证），DSM 实测 256KB ≤ 448KB 真
48-way。小 shape（n<65536）/非连续/非 fp16-bf16-fp32/超 int32 域回退公共
pointwise_dynamic 实现（v2 已验证小 shape 1.5x native）。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def square_flat_kernel(X, Y, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(X + off, mask=mask).to(tl.float32)
        y = x * x
        tl.store(Y + off, y.to(Y.dtype.element_ty), mask=mask)


def _choose_block(n_total, itemsize):
    # BLOCK ladder swept on this kernel body (see module docstring); mid
    # buckets follow the validated family ladder (rad2deg/special_exp2).
    if n_total <= 65536:
        return 8192
    if n_total <= 524288:
        return 32768
    if n_total > (1 << 28) and itemsize <= 2:
        return 131072
    return 65536


def _launch_flat(inp, out):
    n_total = inp.numel()
    BLOCK = _choose_block(n_total, inp.element_size())
    grid_size = min(triton.cdiv(n_total, BLOCK), NUM_SIPS * 2)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared nodes);
    # retrying the launch recompiles and succeeds (atan2_/sinh/rad2deg
    # precedents for the same toolchain family).
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                square_flat_kernel[(grid_size,)](
                    inp,
                    out,
                    n_total,
                    BLOCK=BLOCK,
                    num_warps=2,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "square flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def square(A):
    logger.debug("GEMS_ENFLAME SQUARE")
    n = A.numel()
    if (
        n < 65536
        or not A.is_contiguous()
        or A.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or n >= (1 << 31)
    ):
        from flag_gems.ops.square import square as _public

        return _public(A)
    out = torch.empty_like(A)
    _launch_flat(A, out)
    return out
