import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

# x ** p is decomposed as exp2(p * log2(x)): tl_extra_shim.pow is too heavy
exp2 = tl_extra_shim.exp2
log2 = tl_extra_shim.log2

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def pairwise_distance_general_kernel(
    x1_ptr, x2_ptr, out_ptr, N, D, eps, p, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    p = p.to(acc_dtype)
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    row_mask = pid < N
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)[None, :]
        col_mask = cols < D
        mask = row_mask & col_mask
        a = tl.load(x1_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        b = tl.load(x2_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        diff = tl.abs(a - b + eps)
        acc += tl.sum(tl.where(mask, exp2(p * log2(diff)), 0.0), axis=1)
    dist = exp2((1.0 / p) * log2(acc))
    tl.store(out_ptr + pid, dist[:, None], row_mask)


@libentry()
@triton.jit
def pairwise_distance_p2_kernel(
    x1_ptr, x2_ptr, out_ptr, N, D, eps, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    row_mask = pid < N
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)[None, :]
        col_mask = cols < D
        mask = row_mask & col_mask
        a = tl.load(x1_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        b = tl.load(x2_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        diff = tl.abs(a - b + eps)
        acc += tl.sum(tl.where(mask, (diff * diff), 0.0), axis=1)

    dist = tl.sqrt(acc)
    tl.store(out_ptr + pid, dist[:, None], row_mask)


@libentry()
@triton.jit
def pairwise_distance_p1_kernel(
    x1_ptr, x2_ptr, out_ptr, N, D, eps, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    row_mask = pid < N
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)[None, :]
        col_mask = cols < D
        mask = row_mask & col_mask
        a = tl.load(x1_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        b = tl.load(x2_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        diff = tl.abs(a - b + eps)
        acc += tl.sum(tl.where(mask, diff, 0.0), axis=1)

    tl.store(out_ptr + pid, acc[:, None], row_mask)


@libentry()
@triton.jit
def pairwise_distance_p0_kernel(
    x1_ptr, x2_ptr, out_ptr, N, D, eps, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    row_mask = pid < N
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)[None, :]
        col_mask = cols < D
        mask = row_mask & col_mask
        a = tl.load(x1_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        b = tl.load(x2_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        diff = tl.abs(a - b + eps)
        acc += tl.sum(tl.where(mask, (diff != 0).to(acc_dtype), 0.0), axis=1)

    tl.store(out_ptr + pid, acc[:, None], row_mask)


@libentry()
@triton.jit
def pairwise_distance_max_kernel(
    x1_ptr, x2_ptr, out_ptr, N, D, eps, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = pid < N
    max_val = tl.full([BLOCK_M], -float("inf"), acc_dtype)
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)[None, :]
        col_mask = cols < D
        mask = row_mask & col_mask
        a = tl.load(x1_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        b = tl.load(x2_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        diff = tl.abs(a - b + eps)
        max_val = tl.maximum(
            max_val, tl.max(tl.where(mask, diff, -float("inf")), axis=1)
        )
    tl.store(out_ptr + pid, max_val[:, None], row_mask)


@libentry()
@triton.jit
def pairwise_distance_min_kernel(
    x1_ptr, x2_ptr, out_ptr, N, D, eps, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    min_val = tl.full([BLOCK_M], float("inf"), acc_dtype)
    row_mask = pid < N
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)[None, :]
        col_mask = cols < D
        mask = row_mask & col_mask
        a = tl.load(x1_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        b = tl.load(x2_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        diff = tl.abs(a - b + eps)
        min_val = tl.minimum(
            min_val, tl.min(tl.where(mask, diff, float("inf")), axis=1)
        )
    tl.store(out_ptr + pid, min_val[:, None], row_mask)


@libentry()
@triton.jit
def pairwise_distance_p2_kernel_1(
    x1_ptr, x2_ptr, mid_ptr, D, eps, MID_SIZE, BLOCK_SIZE: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base = pid_n * D
    mask = offset < D
    a = tl.load(x1_ptr + base + offset, mask=mask, other=0.0).to(acc_dtype)
    b = tl.load(x2_ptr + base + offset, mask=mask, other=0.0).to(acc_dtype)
    diff = tl.abs(a - b + eps)
    mid = tl.sum(tl.where(mask, diff * diff, 0.0))
    tl.store(mid_ptr + pid_n * MID_SIZE + pid_d, mid)


@libentry()
@triton.jit
def pairwise_distance_p2_kernel_2(mid_ptr, out_ptr, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < MID_SIZE
    mid = tl.load(mid_ptr + pid * MID_SIZE + offset, mask=mask, other=0.0)
    sum = tl.sqrt(tl.sum(mid))

    tl.store(out_ptr + pid, sum)


@libentry()
@triton.jit
def pairwise_distance_p1_kernel_1(
    x1_ptr, x2_ptr, mid_ptr, D, eps, MID_SIZE, BLOCK_SIZE: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base = pid_n * D
    mask = offset < D
    a = tl.load(x1_ptr + base + offset, mask=mask, other=0.0).to(acc_dtype)
    b = tl.load(x2_ptr + base + offset, mask=mask, other=0.0).to(acc_dtype)
    diff = tl.abs(a - b + eps)
    mid = tl.sum(tl.where(mask, diff, 0.0))
    tl.store(mid_ptr + pid_n * MID_SIZE + pid_d, mid)


@libentry()
@triton.jit
def pairwise_distance_p1_kernel_2(mid_ptr, out_ptr, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < MID_SIZE
    mid = tl.load(mid_ptr + pid * MID_SIZE + offset, mask=mask, other=0.0)
    sum = tl.sum(mid)

    tl.store(out_ptr + pid, sum)


@libentry()
@triton.jit
def pairwise_distance_p0_kernel_1(
    x1_ptr, x2_ptr, mid_ptr, D, eps, MID_SIZE, BLOCK_SIZE: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base = pid_n * D
    mask = offset < D
    a = tl.load(x1_ptr + base + offset, mask=mask, other=0.0).to(acc_dtype)
    b = tl.load(x2_ptr + base + offset, mask=mask, other=0.0).to(acc_dtype)
    diff = tl.abs(a - b + eps)
    mid = tl.sum(tl.where(mask, (diff != 0).to(acc_dtype), 0.0))
    tl.store(mid_ptr + pid_n * MID_SIZE + pid_d, mid)


@libentry()
@triton.jit
def pairwise_distance_p0_kernel_2(mid_ptr, out_ptr, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < MID_SIZE
    mid = tl.load(mid_ptr + pid * MID_SIZE + offset, mask=mask, other=0.0)
    sum = tl.sum(mid)

    tl.store(out_ptr + pid, sum)


@libentry()
@triton.jit
def pairwise_distance_general_kernel_1(
    x1_ptr, x2_ptr, mid_ptr, D, eps, p, MID_SIZE, BLOCK_SIZE: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    p = p.to(acc_dtype)
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base = pid_n * D
    mask = offset < D
    a = tl.load(x1_ptr + base + offset, mask=mask, other=0.0).to(acc_dtype)
    b = tl.load(x2_ptr + base + offset, mask=mask, other=0.0).to(acc_dtype)
    diff = tl.abs(a - b + eps)
    mid = tl.sum(tl.where(mask, exp2(p * log2(diff)), 0.0))
    tl.store(mid_ptr + pid_n * MID_SIZE + pid_d, mid)


@libentry()
@triton.jit
def pairwise_distance_general_kernel_2(
    mid_ptr, out_ptr, p, MID_SIZE, BLOCK_SIZE: tl.constexpr
):
    acc_dtype = tl.float64 if mid_ptr.type.element_ty == tl.float64 else tl.float32
    p = p.to(acc_dtype)
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < MID_SIZE
    mid = tl.load(mid_ptr + pid * MID_SIZE + offset, mask=mask, other=0.0)
    sum = exp2((1 / p) * log2(tl.sum(mid)))

    tl.store(out_ptr + pid, sum)


@libentry()
@triton.jit
def pairwise_distance_max_kernel_1(
    x1_ptr, x2_ptr, mid_ptr, D, eps, MID_SIZE, BLOCK_SIZE: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base = pid_n * D
    mask = offset < D
    a = tl.load(x1_ptr + base + offset, mask=mask, other=0.0).to(acc_dtype)
    b = tl.load(x2_ptr + base + offset, mask=mask, other=0.0).to(acc_dtype)
    diff = tl.abs(a - b + eps)
    mid = tl.max(tl.where(mask, diff, -float("inf")))
    tl.store(mid_ptr + pid_n * MID_SIZE + pid_d, mid)


@libentry()
@triton.jit
def pairwise_distance_max_kernel_2(
    mid_ptr, out_ptr, MID_SIZE, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < MID_SIZE
    mid = tl.load(mid_ptr + pid * MID_SIZE + offset, mask=mask, other=0.0)
    max_val = tl.max(tl.where(mask, mid, -float("inf")))

    tl.store(out_ptr + pid, max_val)


@libentry()
@triton.jit
def pairwise_distance_min_kernel_1(
    x1_ptr, x2_ptr, mid_ptr, D, eps, MID_SIZE, BLOCK_SIZE: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base = pid_n * D
    mask = offset < D
    a = tl.load(x1_ptr + base + offset, mask=mask, other=0.0).to(acc_dtype)
    b = tl.load(x2_ptr + base + offset, mask=mask, other=0.0).to(acc_dtype)
    diff = tl.abs(a - b + eps)
    mid = tl.min(tl.where(mask, diff, float("inf")))
    tl.store(mid_ptr + pid_n * MID_SIZE + pid_d, mid)


@libentry()
@triton.jit
def pairwise_distance_min_kernel_2(
    mid_ptr, out_ptr, MID_SIZE, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < MID_SIZE
    mid = tl.load(mid_ptr + pid * MID_SIZE + offset, mask=mask, other=0.0)
    min_val = tl.min(tl.where(mask, mid, float("inf")))

    tl.store(out_ptr + pid, min_val)


def _single_config(D, heavy=False):
    """Explicit per-D tile (no autotune).

    Light kernels (p in {0,1,2,inf,-inf}) keep BLOCK_D aligned to D so loads
    have no masked waste; BLOCK_M fills the ~8192-elem per-tile budget. The
    exp2/log2 general-p kernel uses a 4096-elem budget with BLOCK_D <= 2048:
    that reproduces the shared op's autotuned tile for large D (e.g. (2, 2048)
    at D=65536), which is the config that passes the fp32 1.3e-6-rtol accuracy
    tests, and it keeps UB within budget.
    """
    if heavy:
        bd = min(triton.next_power_of_2(D), 2048)
        bm = max(1, min(512, 4096 // bd))
        return bm, bd
    bd = min(triton.next_power_of_2(D), 4096)
    bm = max(1, min(512, 8192 // bd))
    return bm, bd


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    logger.debug("GEMS_ASCEND PAIRWISE_DISTANCE")
    # Only broadcast/contiguous when needed (same shape + contiguous is the common
    # case; unconditional broadcast_tensors + contiguous materializes a copy).
    if x1.shape != x2.shape:
        x1, x2 = torch.broadcast_tensors(x1, x2)
    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    D = x1.shape[-1]

    # Empty feature dim: torch returns 0 for finite p; inf/-inf have no identity
    # element over an empty reduction. Short-circuit (also avoids ZeroDivisionError
    # in the split-K plumbing when BLOCK_SIZE == 0).
    if D == 0:
        if p == float("inf") or p == float("-inf"):
            raise RuntimeError(
                "pairwise_distance cannot compute the inf/-inf norm on an empty "
                "reduction dimension (no identity element)"
            )
        out = torch.zeros(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
        if keepdim:
            out = out.unsqueeze(-1)
        return out

    N = x1.numel() // D
    out = torch.empty(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
    if keepdim:
        out = out.unsqueeze(-1)
    mid_dtype = torch.float64 if x1.dtype == torch.float64 else torch.float32

    with torch_device_fn.device(x1.device):
        BLOCK_SIZE = min(triton.next_power_of_2(D), 4096)
        MID_SIZE = triton.cdiv(D, BLOCK_SIZE)
        # Split-K only pays off when the row is long enough to need many chunks:
        # on Ascend each extra launch costs ~0.1ms, so splitting e.g. D=65536
        # (MID_SIZE=16) is slower than the single kernel. Keep the shared op's
        # small-N guard but require a large chunk count (D ~>= 1M).
        use_split_k = (N <= 128) and (MID_SIZE >= 256)

        if use_split_k:
            BLOCK_MID = triton.next_power_of_2(MID_SIZE)
            mid = torch.empty((N, MID_SIZE), device=x1.device, dtype=mid_dtype)
            if p == 2.0:
                pairwise_distance_p2_kernel_1[(N, MID_SIZE)](
                    x1, x2, mid, D, eps, MID_SIZE, BLOCK_SIZE
                )
                pairwise_distance_p2_kernel_2[(N,)](mid, out, MID_SIZE, BLOCK_MID)
            elif p == 1.0:
                pairwise_distance_p1_kernel_1[(N, MID_SIZE)](
                    x1, x2, mid, D, eps, MID_SIZE, BLOCK_SIZE
                )
                pairwise_distance_p1_kernel_2[(N,)](mid, out, MID_SIZE, BLOCK_MID)
            elif p == 0.0:
                pairwise_distance_p0_kernel_1[(N, MID_SIZE)](
                    x1, x2, mid, D, eps, MID_SIZE, BLOCK_SIZE
                )
                pairwise_distance_p0_kernel_2[(N,)](mid, out, MID_SIZE, BLOCK_MID)
            elif p == float("inf"):
                pairwise_distance_max_kernel_1[(N, MID_SIZE)](
                    x1, x2, mid, D, eps, MID_SIZE, BLOCK_SIZE
                )
                pairwise_distance_max_kernel_2[(N,)](mid, out, MID_SIZE, BLOCK_MID)
            elif p == float("-inf"):
                pairwise_distance_min_kernel_1[(N, MID_SIZE)](
                    x1, x2, mid, D, eps, MID_SIZE, BLOCK_SIZE
                )
                pairwise_distance_min_kernel_2[(N,)](mid, out, MID_SIZE, BLOCK_MID)
            else:
                pairwise_distance_general_kernel_1[(N, MID_SIZE)](
                    x1, x2, mid, D, eps, p, MID_SIZE, BLOCK_SIZE
                )
                pairwise_distance_general_kernel_2[(N,)](
                    mid, out, p, MID_SIZE, BLOCK_MID
                )
            return out

        is_general = p not in (2.0, 1.0, 0.0, float("inf"), float("-inf"))
        bm, bd = _single_config(D, heavy=is_general)
        grid = (triton.cdiv(N, bm),)
        if p == 2.0:
            pairwise_distance_p2_kernel[grid](
                x1, x2, out, N, D, eps, BLOCK_M=bm, BLOCK_D=bd
            )
        elif p == 1.0:
            pairwise_distance_p1_kernel[grid](
                x1, x2, out, N, D, eps, BLOCK_M=bm, BLOCK_D=bd
            )
        elif p == 0.0:
            pairwise_distance_p0_kernel[grid](
                x1, x2, out, N, D, eps, BLOCK_M=bm, BLOCK_D=bd
            )
        elif p == float("inf"):
            pairwise_distance_max_kernel[grid](
                x1, x2, out, N, D, eps, BLOCK_M=bm, BLOCK_D=bd
            )
        elif p == float("-inf"):
            pairwise_distance_min_kernel[grid](
                x1, x2, out, N, D, eps, BLOCK_M=bm, BLOCK_D=bd
            )
        else:
            pairwise_distance_general_kernel[grid](
                x1, x2, out, N, D, eps, p, BLOCK_M=bm, BLOCK_D=bd
            )

    return out
