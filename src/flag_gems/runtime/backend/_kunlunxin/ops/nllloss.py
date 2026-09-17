import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .sum import sum as xpu_sum
from .zero import zero_
from .zeros_like import zeros_like as xpu_zeros_like

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_wgt_tgt_ptr,
    ignore_index,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
    PADDED: tl.constexpr = False,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    assert (tgt == ignore_index) or (tgt >= 0 and tgt < C), "Invalid target value"
    ignore_mask = not (tgt == ignore_index) and mask_n

    if wgt_ptr is None:
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    inp_tgt_ptrs = inp_ptr + offsets_n * C + tgt
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    if PADDED:
        tl.store(out_ptr + offsets_n, out)
        tl.store(ignore_wgt_tgt_ptr + offsets_n, wgt_tgt)
    else:
        tl.store(out_ptr + offsets_n, out, mask=mask_n)
        if reduction != 0:
            tl.store(ignore_wgt_tgt_ptr + offsets_n, wgt_tgt, mask=mask_n)


_NLL_REDUCE_TILE = 8192
_NLL_REDUCE_MAX_TILES = 2
_NLL_FWD_BLOCK_SMALL = 256
_NLL_FWD_BLOCK_LARGE = 512
_NLL_FWD_BLOCK_SWITCH = 1024


def _nll_fwd_block(n):
    return _NLL_FWD_BLOCK_SMALL if n <= _NLL_FWD_BLOCK_SWITCH else _NLL_FWD_BLOCK_LARGE


@libentry()
@triton.jit
def nll_loss_reduce_kernel(
    out_ptr,
    wgt_ptr,
    total_out_ptr,
    total_wgt_ptr,
    MEAN: tl.constexpr,
    NTILES: tl.constexpr,
    TL: tl.constexpr,
):
    total_o = tl.zeros([], dtype=tl.float32)
    total_wgt = tl.zeros([], dtype=tl.float32)
    for i in tl.static_range(NTILES):
        off = i * TL + tl.arange(0, TL)
        o = tl.load(out_ptr + off).to(tl.float32)
        w = tl.load(wgt_ptr + off).to(tl.float32)
        total_o += tl.sum(o)
        total_wgt += tl.sum(w)

    if MEAN:
        res = total_o / total_wgt
    else:
        res = total_o
    tl.store(total_out_ptr, res.to(total_out_ptr.dtype.element_ty))
    tl.store(total_wgt_ptr, total_wgt.to(total_wgt_ptr.dtype.element_ty))


_NLL2D_PARTIAL_PROG_TILES = 4


def _nll2d_partial_config(ntiles):
    """`(nprog, tpp)` with `tpp` the largest divisor of `ntiles` at most
    `_NLL2D_PARTIAL_PROG_TILES` (so `nprog * tpp == ntiles` exactly): no
    program body is unrolled past the 4-tile fault cap and no program ever
    touches a tile outside `[0, ntiles)`."""
    for tpp in (_NLL2D_PARTIAL_PROG_TILES, 2, 1):
        if ntiles % tpp == 0:
            return ntiles // tpp, tpp
    raise RuntimeError("unreachable")


@libentry()
@triton.jit
def nll_loss2d_partial_reduce_kernel(
    out_ptr,
    wgt_ptr,
    pout_ptr,
    pwgt_ptr,
    TPP: tl.constexpr,
    TL: tl.constexpr,
):
    pid = tl.program_id(0)
    total_o = tl.zeros([], dtype=tl.float32)
    total_w = tl.zeros([], dtype=tl.float32)
    for j in tl.static_range(TPP):
        off = (pid * TPP + j) * TL + tl.arange(0, TL)
        o = tl.load(out_ptr + off).to(tl.float32)
        w = tl.load(wgt_ptr + off).to(tl.float32)
        total_o += tl.sum(o)
        total_w += tl.sum(w)
    tl.store(pout_ptr + pid, total_o)
    tl.store(pwgt_ptr + pid, total_w)


@libentry()
@triton.jit
def nll_loss2d_finalize_kernel(
    pout_ptr,
    pwgt_ptr,
    total_out_ptr,
    total_wgt_ptr,
    MEAN: tl.constexpr,
    NP: tl.constexpr,
    nprog,
):
    off = tl.arange(0, NP)
    mask = off < nprog
    total_o = tl.sum(tl.load(pout_ptr + off, mask=mask, other=0).to(tl.float32))
    total_w = tl.sum(tl.load(pwgt_ptr + off, mask=mask, other=0).to(tl.float32))
    if MEAN:
        res = total_o / total_w
    else:
        res = total_o
    tl.store(total_out_ptr, res.to(total_out_ptr.dtype.element_ty))
    tl.store(total_wgt_ptr, total_w.to(total_wgt_ptr.dtype.element_ty))


@libentry()
@triton.jit
def nll_loss_scalar_mean_kernel(num_ptr, den_ptr, out_ptr, tw_ptr):
    num = tl.load(num_ptr).to(tl.float32)
    den = tl.load(den_ptr).to(tl.float32)
    tl.store(out_ptr, (num / den).to(out_ptr.dtype.element_ty))
    tl.store(tw_ptr, den.to(tw_ptr.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_n

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offsets_n
        out_grad = tl.load(out_grad_ptrs, mask=mask_n, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)
    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1

    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0)
    inp_grad_ptrs = inp_grad_ptr + offsets_n * C + tgt
    tl.store(inp_grad_ptrs, inp_grad, mask=ignore_mask)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_wgt_tgt_ptr,
    ignore_index,
    N,
    C,
    D,
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = 128,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)
    offset_d = offset_nd % D
    offset_n = offset_nd // D

    mask_block = offset_nd < N * D

    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    assert (tgt == ignore_index) or (tgt >= 0 and tgt < C), "Invalid target value"
    ignore_mask = not (tgt == ignore_index) and mask_block

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    inp_tgt_ptrs = inp_ptr + offset_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    out_ptrs = out_ptr + offset_n * D + offset_d
    tl.store(out_ptrs, out, mask=mask_block)

    if reduction != 0:
        ignore_wgt_tgt_ptrs = ignore_wgt_tgt_ptr + offset_n * D + offset_d
        tl.store(ignore_wgt_tgt_ptrs, wgt_tgt, mask=mask_block)


_NLL2D_MIN_BLOCK_D = 128
_NLL2D_MAX_BLOCK_D = 512


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_forward_tiled_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_wgt_tgt_ptr,
    ignore_index,
    C,
    reduction: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILE_N: tl.constexpr = 1,
):
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = (pid_n * TILE_N + tl.arange(0, TILE_N))[:, None]
    cols = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D))[None, :]
    flat = rows * D + cols

    tgt = tl.load(tgt_ptr + flat)
    assert (tgt == ignore_index) or (tgt >= 0 and tgt < C), "Invalid target value"
    ignore_mask = tgt != ignore_index

    if wgt_ptr is None:
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    inp_tgt_ptrs = inp_ptr + rows * (C * D) + tgt * D + cols
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    tl.store(out_ptr + flat, out)
    if reduction != 0:
        tl.store(ignore_wgt_tgt_ptr + flat, wgt_tgt)


_NLL2D_FLAT_BLOCKS = (2048, 512, 256, 128)


def _nll2d_flat_cap(M):
    """Measured best `BLOCK_ND` band for the wide flat kernel.

    Wrapper-level `do_bench` minima over 3 interleaved rounds, `weight=None`
    (the weight-gather cells are bimodal and unusable for tuning), block cap
    swept 0(=HEAD)/128/256/512/1024/2048/4096/8192, fp16 / fp32 in us:

      M=2048    39.6/23.9  20.7/15.2  13.9/11.7  *12.6/11.3*  14.3/14.5  20.6/21.1  ...
      M=8192    68.2/55.0  35.6/29.7  26.0/20.5   19.2/15.1   [14.9/14.8] 21.8/21.6  35.5/35.4  61.9/61.3
      M=131072  425/375    223/194    150/135     113/98       96.8/90.3 *89.9/88.8* 94.9/94.4  117/116

    The optimum grows with `M` (more parallelism needed) but a single tile that
    covers all of `M` is always bad (M=8192 @ 8192 -> 61 us vs 14.9 us @ 1024).
    The M=8192 optimum (1024, in brackets) is *not* usable - see
    `_NLL2D_FLAT_BLOCKS` - so 512 is taken there instead.
    """
    return 512 if M <= 65536 else 2048


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_forward_flat_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_wgt_tgt_ptr,
    ignore_index,
    C,
    reduction: tl.constexpr,
    D: tl.constexpr,
    BLOCK_ND: tl.constexpr,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)

    tgt = tl.load(tgt_ptr + offset_nd)
    assert (tgt == ignore_index) or (tgt >= 0 and tgt < C), "Invalid target value"
    ignore_mask = tgt != ignore_index

    if wgt_ptr is None:
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    offset_d = offset_nd % D
    offset_n = offset_nd // D
    inp_tgt_ptrs = inp_ptr + offset_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    tl.store(out_ptr + offset_nd, out)
    if reduction != 0:
        tl.store(ignore_wgt_tgt_ptr + offset_nd, wgt_tgt)


def _nll2d_flat_block(M):
    """Widest verified-safe block that divides `M` and fits `_nll2d_flat_cap(M)`.

    `None` means "no unmasked wide tiling possible"; the caller then keeps the
    original 128-wide masked flat kernel.
    """
    cap = _nll2d_flat_cap(M)
    for block in _NLL2D_FLAT_BLOCKS:
        if block <= cap and M % block == 0:
            return block
    return None


def _nll2d_block_d(D):
    """Largest power-of-two `BLOCK_D` in [128, 512] that exactly divides `D`.

    Returns `None` when no such value exists (odd / small trailing dims), in
    which case `nll_loss2d_forward` keeps the original flat kernel.
    """
    block_d = 1
    cap = min(D, _NLL2D_MAX_BLOCK_D)
    while block_d * 2 <= cap and D % (block_d * 2) == 0:
        block_d *= 2
    return block_d if block_d >= _NLL2D_MIN_BLOCK_D else None


_NLL2D_REDUCE_TILE = 8192
_NLL2D_REDUCE_MAX_TILES = 4


def _nll2d_fused_reduce(M):
    """Return `(ntiles, TL)` if `M` elements can be reduced by one unmasked
    fused launch, else `None`."""
    tl_width = min(_NLL2D_REDUCE_TILE, triton.next_power_of_2(M))
    ntiles = triton.cdiv(M, tl_width)
    if ntiles > _NLL2D_REDUCE_MAX_TILES or ntiles * tl_width != M:
        return None
    return ntiles, tl_width


_NLL2D_EXACT_TILE_CAP = 128


def _nll2d_exact_tiles(M):
    """Return `(ntiles, TL)` if `M` elements tile exactly (no padding), else
    `None`.  This is `_nll2d_fused_reduce` without the single-program cap, so
    the caller can choose between the one-program fused path (ntiles <=
    `_NLL2D_REDUCE_MAX_TILES`), the grid-parallel path (ntiles <=
    `_NLL2D_EXACT_TILE_CAP`) and keep the staged `xpu_sum` tail otherwise.
    Both reduced paths require fully unmasked tiles, i.e. `M % TL == 0`."""
    tl_width = min(_NLL2D_REDUCE_TILE, triton.next_power_of_2(M))
    ntiles = triton.cdiv(M, tl_width)
    if ntiles > _NLL2D_EXACT_TILE_CAP or ntiles * tl_width != M:
        return None
    return ntiles, tl_width


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    D,
    reduction: tl.constexpr = 1,
    BLOCK_ND: tl.constexpr = 128,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)
    offset_d = offset_nd % D
    offset_n = offset_nd // D

    mask_block = offset_nd < N * D

    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_block

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offset_n * D + offset_d
        out_grad = tl.load(out_grad_ptrs, mask=mask_block, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1
    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0)
    inp_grad_ptrs = inp_grad_ptr + offset_n * C * D + tgt * D + offset_d
    tl.store(inp_grad_ptrs, inp_grad, mask=ignore_mask)


_NLL2D_BWD_FLAT_BLOCKS = (4096, 1024, 512, 256, 128)


def _nll2d_bwd_flat_cap(M):
    """Measured best `BLOCK_ND` band for the wide flat backward scatter.

    Kernel-only `do_bench` minima, summed over 3 dtypes x weight{Y,N} x
    reduction{none,mean,sum} (18 cells per row, us):

      N*D      HEAD(128)  128    256    512    1024   2048   4096   8192
      2048        255     240    220    220    ---    ---     -      -
      8192        515     ---    ---    385    370    ---     -      -
      16384      1134     ---    ---    ---    900    925     -      -
      131072     9383     ---    ---    ---    ---   7765   7577   8000+

    A single tile covering all of `N*D` is always bad, and the optimum grows
    with `N*D`; the weight-gather cells are flat in the width (the discrete
    gather dominates them) so the band is set by the `weight=None` cells.
    """
    if M <= 2048:
        return 512
    if M <= 65536:
        return 1024
    return 4096


def _nll2d_bwd_flat_block(M):
    """Widest whitelisted block that divides `M` and fits `_nll2d_bwd_flat_cap`.

    `None` means "no unmasked wide tiling possible"; the caller then keeps the
    original 128-wide masked flat kernel byte-for-byte.
    """
    cap = _nll2d_bwd_flat_cap(M)
    for block in _NLL2D_BWD_FLAT_BLOCKS:
        if block <= cap and M % block == 0:
            return block
    return None


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_backward_flat_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    C,
    reduction: tl.constexpr,
    D: tl.constexpr,
    BLOCK_ND: tl.constexpr,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)

    tgt = tl.load(tgt_ptr + offset_nd)
    ignore_mask = tgt != ignore_index
    safe_tgt = tl.where(ignore_mask, tgt, 0)

    if reduction == 0:
        out_grad = tl.load(out_grad_ptr + offset_nd).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    if wgt_ptr is None:
        wgt_tgt = tl.where(ignore_mask, 1.0, 0.0)
    else:
        wgt_tgt = tl.load(wgt_ptr + safe_tgt).to(tl.float32)

    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1.0

    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0.0)
    offset_d = offset_nd % D
    offset_n = offset_nd // D
    inp_grad_ptrs = inp_grad_ptr + offset_n * (C * D) + safe_tgt * D + offset_d
    tl.store(inp_grad_ptrs, inp_grad)


def nll_loss_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS_KUNLUNXIN NLL_LOSS_FWD")
    assert self.ndim <= 2, "Invalid input ndim"

    if self.numel() == 0:
        if reduction == 0:
            loss = torch.empty(target.shape, dtype=self.dtype, device=self.device)
        elif reduction == 1:
            loss = torch.full((), float("nan"), dtype=self.dtype, device=self.device)
        else:
            loss = torch.zeros((), dtype=self.dtype, device=self.device)
        total_weight = torch.zeros((), dtype=self.dtype, device=self.device)
        return loss, total_weight

    shape = list(target.shape)
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]
    assert target.numel() == N, "Invalid target size"

    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    BLOCK_N = _nll_fwd_block(N)
    fused = False
    if reduction != 0:
        TL = min(_NLL_REDUCE_TILE, triton.next_power_of_2(N))
        ntiles = triton.cdiv(N, TL)
        fused = ntiles <= _NLL_REDUCE_MAX_TILES

    if fused:
        pad_n = triton.cdiv(ntiles * TL, BLOCK_N) * BLOCK_N
        out = torch.empty(pad_n, dtype=self.dtype, device=self.device)
        ignore_weight_tgt = torch.empty(pad_n, dtype=self.dtype, device=self.device)
        n_blocks = pad_n // BLOCK_N
    else:
        out = torch.empty(shape, dtype=self.dtype, device=self.device)
        ignore_weight_tgt = None
        if reduction != 0:
            ignore_weight_tgt = torch.empty(
                target.shape, dtype=self.dtype, device=self.device
            )
        n_blocks = triton.cdiv(N, BLOCK_N)

    with torch_device_fn.device(self.device):
        nll_loss_forward_kernel[(n_blocks, 1, 1)](
            self,
            target,
            weight,
            out,
            ignore_weight_tgt,
            ignore_index,
            N,
            C,
            reduction,
            BLOCK_N,
            fused,
            is_use_mask_zero=True,
        )

    if reduction == 0:
        return out, torch.zeros([], dtype=self.dtype, device=self.device)

    if fused:
        output = torch.empty([], dtype=self.dtype, device=self.device)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
        with torch_device_fn.device(self.device):
            nll_loss_reduce_kernel[(1, 1, 1)](
                out,
                ignore_weight_tgt,
                output,
                total_weight,
                reduction == 1,
                ntiles,
                TL,
            )
        return output, total_weight

    if reduction == 1:
        total_out = xpu_sum(out)
        total_weight = xpu_sum(ignore_weight_tgt).to(self.dtype)
        output = (total_out / total_weight).to(self.dtype)
    else:
        total_out = xpu_sum(out)
        output = total_out.to(self.dtype)
        total_weight = xpu_sum(ignore_weight_tgt).to(self.dtype)

    return output, total_weight


def nll_loss_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    logger.debug("GEMS_KUNLUNXIN NLL_LOSS_BWD")
    if self.numel() == 0:
        return torch.empty_like(self)
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    if self.is_contiguous():
        grad_input = xpu_zeros_like(self)
    else:
        grad_input = torch.empty_like(self).contiguous()
        zero_(grad_input)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    with torch_device_fn.device(self.device):
        nll_loss_backward_kernel[grid](
            grad_output,
            target,
            weight,
            grad_input,
            ignore_index,
            total_weight,
            N,
            C,
            reduction,
        )

    return grad_input


def nll_loss2d_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS_KUNLUNXIN NLL_LOSS2D_FWD")
    assert self.ndim >= 3, "Invalid input ndim"

    if self.numel() == 0:
        if reduction == 0:
            loss = torch.empty(target.shape, dtype=self.dtype, device=self.device)
        elif reduction == 1:
            loss = torch.full((), float("nan"), dtype=self.dtype, device=self.device)
        else:
            loss = torch.zeros((), dtype=self.dtype, device=self.device)
        total_weight = torch.zeros((), dtype=self.dtype, device=self.device)
        return loss, total_weight

    N, C = self.shape[0], self.shape[1]
    D = self.numel() // (N * C)
    assert target.numel() == N * D, "Invalid target size"

    target_orig_shape = target.shape
    self_flat = self.reshape(N, C, D).contiguous()
    target_flat = target.reshape(N, D).contiguous()
    weight = None if weight is None else weight.contiguous()

    out = torch.empty((N, D), dtype=self.dtype, device=self.device)
    exact = _nll2d_exact_tiles(N * D) if reduction != 0 else None
    ignore_weight_tgt = None
    if reduction != 0:
        ignore_weight_tgt = torch.empty((N, D), dtype=self.dtype, device=self.device)

    block_d = _nll2d_block_d(D)
    flat_block = None if block_d is not None else _nll2d_flat_block(N * D)
    with torch_device_fn.device(self.device):
        if flat_block is not None:
            nll_loss2d_forward_flat_kernel[((N * D) // flat_block, 1, 1)](
                self_flat,
                target_flat,
                weight,
                out,
                ignore_weight_tgt,
                ignore_index,
                C,
                reduction,
                D,
                flat_block,
                is_use_mask_zero=True,
            )
        elif block_d is None:
            grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
            nll_loss2d_forward_kernel[grid](
                self_flat,
                target_flat,
                weight,
                out,
                ignore_weight_tgt,
                ignore_index,
                N,
                C,
                D,
                reduction,
                is_use_mask_zero=True,
            )
        else:
            nll_loss2d_forward_tiled_kernel[(D // block_d, N, 1)](
                self_flat,
                target_flat,
                weight,
                out,
                ignore_weight_tgt,
                ignore_index,
                C,
                reduction,
                D,
                block_d,
                is_use_mask_zero=True,
            )

    if reduction == 0:
        output = out.reshape(target_orig_shape)
        total_weight = torch.zeros([], dtype=self.dtype, device=self.device)
        return output, total_weight

    if exact is not None:
        ntiles, tl_width = exact
        output = torch.empty([], dtype=self.dtype, device=self.device)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
        wgt_buf = ignore_weight_tgt
        with torch_device_fn.device(self.device):
            if ntiles <= _NLL2D_REDUCE_MAX_TILES:
                nll_loss_reduce_kernel[(1, 1, 1)](
                    out,
                    wgt_buf,
                    output,
                    total_weight,
                    reduction == 1,
                    ntiles,
                    tl_width,
                )
            else:
                nprog, tpp = _nll2d_partial_config(ntiles)
                pout = torch.empty((nprog,), dtype=torch.float32, device=self.device)
                pwgt = torch.empty((nprog,), dtype=torch.float32, device=self.device)
                nll_loss2d_partial_reduce_kernel[(nprog, 1, 1)](
                    out,
                    wgt_buf,
                    pout,
                    pwgt,
                    tpp,
                    tl_width,
                )
                nll_loss2d_finalize_kernel[(1, 1, 1)](
                    pout,
                    pwgt,
                    output,
                    total_weight,
                    reduction == 1,
                    triton.next_power_of_2(nprog),
                    nprog,
                )
        return output, total_weight

    if reduction == 1:
        acc_dtype = (
            torch.float32
            if self.dtype in (torch.float16, torch.bfloat16, torch.float32)
            else None
        )
        total_out = xpu_sum(out, dtype=acc_dtype)
        if weight is None and not (0 <= ignore_index < C):
            exact_count = float(N * D)
            output = (total_out / exact_count).to(self.dtype)
            count_value = (
                exact_count
                if exact_count <= torch.finfo(self.dtype).max
                else float("inf")
            )
            total_weight = torch.full(
                [], count_value, dtype=self.dtype, device=self.device
            )
        else:
            total_weight_acc = xpu_sum(ignore_weight_tgt, dtype=acc_dtype)
            output = torch.empty([], dtype=self.dtype, device=self.device)
            total_weight = torch.empty([], dtype=self.dtype, device=self.device)
            with torch_device_fn.device(self.device):
                nll_loss_scalar_mean_kernel[(1, 1, 1)](
                    total_out,
                    total_weight_acc,
                    output,
                    total_weight,
                )
    else:
        total_out = xpu_sum(out)
        output = total_out.to(self.dtype)
        total_weight = torch.zeros([], dtype=self.dtype, device=self.device)

    return output, total_weight


def nll_loss2d_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    logger.debug("GEMS_KUNLUNXIN NLL_LOSS2D_BWD")
    if self.numel() == 0:
        return torch.empty_like(self)
    N, C = self.shape[0], self.shape[1]
    D = self.numel() // (N * C)

    grad_output = grad_output.contiguous()
    target_flat = target.reshape(N, D).contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.empty_like(self).contiguous()
    zero_(grad_input)

    flat_block = _nll2d_bwd_flat_block(N * D)
    with torch_device_fn.device(self.device):
        if flat_block is not None:
            nll_loss2d_backward_flat_kernel[((N * D) // flat_block, 1, 1)](
                grad_output,
                target_flat,
                weight,
                grad_input.reshape(N, C, D),
                ignore_index,
                total_weight,
                C,
                reduction,
                D,
                flat_block,
            )
        else:
            grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
            nll_loss2d_backward_kernel[grid](
                grad_output,
                target_flat,
                weight,
                grad_input.reshape(N, C, D),
                ignore_index,
                total_weight,
                N,
                C,
                D,
                reduction,
            )

    return grad_input


def nll_loss_nd_forward(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: int = 1,
    ignore_index: int = -100,
):
    logger.debug("GEMS_KUNLUNXIN NLL LOSS ND FWD")
    if input.numel() == 0:
        if reduction == 0:
            loss = torch.empty(target.shape, dtype=input.dtype, device=input.device)
        elif reduction == 1:
            loss = torch.full((), float("nan"), dtype=input.dtype, device=input.device)
        else:
            loss = torch.zeros((), dtype=input.dtype, device=input.device)
        total_weight = torch.zeros((), dtype=input.dtype, device=input.device)
        return loss, total_weight
    if input.dim() < 3:
        return nll_loss_forward(
            input, target, weight=weight, reduction=reduction, ignore_index=ignore_index
        )

    return nll_loss2d_forward(
        input, target, weight=weight, reduction=reduction, ignore_index=ignore_index
    )


def nll_loss_nd_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction: int = 1,
    ignore_index: int = -100,
    total_weight: torch.Tensor = None,
):
    logger.debug("GEMS_KUNLUNXIN NLL LOSS ND BWD")
    if input.numel() == 0:
        return torch.empty_like(input)
    if input.dim() < 3:
        return nll_loss_backward(
            grad_output,
            input,
            target,
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_index,
            total_weight=total_weight,
        )

    return nll_loss2d_backward(
        grad_output,
        input,
        target,
        weight=weight,
        reduction=reduction,
        ignore_index=ignore_index,
        total_weight=total_weight,
    )


class _NllLoss2dAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, target, weight, reduction, ignore_index):
        output, total_weight = nll_loss2d_forward(
            self, target, weight, reduction, ignore_index
        )
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        ctx.save_for_backward(self, target, weight, total_weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        self, target, weight, total_weight = ctx.saved_tensors
        grad_input = nll_loss2d_backward(
            grad_output,
            self,
            target,
            weight=weight,
            reduction=ctx.reduction,
            ignore_index=ctx.ignore_index,
            total_weight=total_weight,
        )
        return grad_input, None, None, None, None


def nll_loss2d(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS_KUNLUNXIN NLL_LOSS2D")
    return _NllLoss2dAutograd.apply(self, target, weight, reduction, ignore_index)
