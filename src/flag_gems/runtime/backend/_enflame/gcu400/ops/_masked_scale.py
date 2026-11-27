"""_masked_scale enflame v3: flat grid-stride family rewrite (square v3 style).

The stock KernelGen copy (pointwise_dynamic one-tile-per-program) moves
9 bytes/element at only ~740GB/s on GCU400 (1G-elem fp32: 13.0ms; 16.7M:
216us). The validated pointwise family form (square/rad2deg v3, see
gcu400/ops/square.py) fixes the traversal instead: persistent grid-stride
loop over flat blocks, grid capped at NUM_SIPS*2=48, ladder swept on this
kernel body (do_bench kernel median, warmup 1000ms / rep 100ms, bit-exact
vs CPU reference at every config):

  n=2^18: old 6.5us -> B8192w2 5.6us        n=2^22: old 55.8us -> B32768w1 20.8us
  n=2^20: old 16.0us -> B8192w2 10.5us /    n=2^24: old 215.8us -> B32768w1 58.4us
          B32768w1 9.6us                    n=2^26: old 833us -> B32768w1 203.5us
                                             n=2^30: old 13111us -> B32768w1 3104us

Ladder: n < 2^18 stays on the generic path (noop-event floor zone, old impl
already at floor); 2^18 <= n < 2^20 -> BLOCK 8192 / num_warps 2; n >= 2^20 ->
BLOCK 32768 / num_warps 1 (ties B65536w1 at 2^20/1G, wins 2^22-2^26).
Rejected variants (same-window A/B): num_warps 4 at mid/large (-16..-66%),
B8192w1 at 2^22 (-21%), B131072 at 16.7M/1G (ties at best, fp32 DSM load
phase 640KB > 448KB breaks the true 48-way budget). B32768 load phase
x 128KB + mask 32KB = 160KB <= 448KB. Kernel body is byte-identical to the
generic expression (tl.where(m != 0, x * scale, 0.0)), so fast-lane outputs
are bit-exact vs the generic op for every dtype routed through it.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# int32 safety domain: flat offsets are computed as block_id * BLOCK + arange
# in int32; the guard keeps N_total + BLOCK - 1 (the block-count numerator)
# and every offset strictly inside the positive int32 range for BLOCK <= 2^17
# (square family guard, same wrap window).
INT32_SAFE_MAX = (1 << 31) - (1 << 17)
SMALL_N = 1 << 18
MID_N = 1 << 20
FAST_MASK_DTYPES = (torch.bool, torch.uint8, torch.int8)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _masked_scale_flat_kernel(Input, Mask, Out, scale, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        keep = off < N_total
        x = tl.load(Input + off, mask=keep)
        m = tl.load(Mask + off, mask=keep)
        y = tl.where(m != 0, x * scale, 0.0)
        tl.store(Out + off, y, mask=keep)


def _choose_block(n_total):
    if n_total < MID_N:
        return 8192, 2
    return 32768, 1


def _launch_flat(input, mask, out, scale):
    n_total = input.numel()
    BLOCK, num_warps = _choose_block(n_total)
    grid_size = min(triton.cdiv(n_total, BLOCK), NUM_SIPS * 2)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared nodes);
    # retrying the launch recompiles and succeeds (square/atan2_/rad2deg
    # precedents for the same toolchain family).
    for attempt in range(3):
        try:
            with torch_device_fn.device(input.device):
                _masked_scale_flat_kernel[(grid_size,)](
                    input,
                    mask,
                    out,
                    scale,
                    n_total,
                    BLOCK=BLOCK,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "_masked_scale flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _masked_scale(input, mask, scale):
    logger.debug("GEMS_ENFLAME _MASKED_SCALE")
    if not input.is_floating_point():
        raise ValueError(f"Only floating-point dtype is supported, got {input.dtype}")
    n = input.numel()
    if (
        n < SMALL_N
        or input.dtype is not torch.float32
        or mask.dtype not in FAST_MASK_DTYPES
        or not input.is_contiguous()
        or not mask.is_contiguous()
        or mask.shape != input.shape
        or n >= INT32_SAFE_MAX
    ):
        from flag_gems.ops._masked_scale import _masked_scale as _public

        return _public(input, mask, scale)
    out = torch.empty_like(input)
    _launch_flat(input, mask, out, scale)
    return out
