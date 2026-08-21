import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext


@libentry()
@triton.jit
def dist_l2_single_kernel(
    X,
    Y,
    Out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros(
        (BLOCK_SIZE,),
        dtype=tl.float32,
    )

    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets

        mask = idx < N

        x = tl.load(
            X + idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        y = tl.load(
            Y + idx,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        diff = x - y

        acc += tl.where(
            mask,
            diff * diff,
            tl.zeros(
                (BLOCK_SIZE,),
                dtype=tl.float32,
            ),
        )

    result = tl.sum(acc)

    tl.store(
        Out,
        tl.sqrt(result),
    )


@libentry()
@triton.jit
def dist_l2_partial_kernel(
    X,
    Y,
    Partial,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N

    x = tl.load(
        X + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    y = tl.load(
        Y + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    diff = x - y

    result = tl.sum(diff * diff)

    tl.store(
        Partial + pid,
        result,
    )


@libentry()
@triton.jit
def dist_reduce_stage1_kernel(
    Partial,
    Partial2,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N

    x = tl.load(
        Partial + offsets,
        mask=mask,
        other=0.0,
    )

    result = tl.sum(x)

    tl.store(
        Partial2 + pid,
        result,
    )


@libentry()
@triton.jit
def dist_reduce_final_kernel(
    Partial,
    Out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)

    mask = offsets < N

    x = tl.load(
        Partial + offsets,
        mask=mask,
        other=0.0,
    )

    result = tl.sum(x)

    tl.store(
        Out,
        tl.sqrt(result),
    )


def _dist_p2(
    input,
    other,
):
    numel = input.numel()

    out = torch.empty(
        [],
        dtype=torch.float32,
        device=input.device,
    )

    if numel <= 65536:
        BLOCK_SIZE = triton.next_power_of_2(min(numel, 4096))

        dist_l2_single_kernel[(1,)](
            input,
            other,
            out,
            numel,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    else:
        BLOCK_SIZE = 4096

        partial_size = triton.cdiv(
            numel,
            BLOCK_SIZE,
        )

        partial = torch.empty(
            partial_size,
            dtype=torch.float32,
            device=input.device,
        )

        mid_block = 4096

        mid_size = triton.cdiv(
            partial_size,
            mid_block,
        )

        partial2 = torch.empty(
            mid_size,
            dtype=torch.float32,
            device=input.device,
        )

        with torch_device_fn.device(input.device):

            dist_l2_partial_kernel[(partial_size,)](
                input,
                other,
                partial,
                numel,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
            )

            dist_reduce_stage1_kernel[(mid_size,)](
                partial,
                partial2,
                partial_size,
                BLOCK_SIZE=mid_block,
                num_warps=8,
            )

            final_block = triton.next_power_of_2(mid_size)

            dist_reduce_final_kernel[(1,)](
                partial2,
                out,
                mid_size,
                BLOCK_SIZE=final_block,
                num_warps=4,
            )

    return out.to(input.dtype)


def _dist_generic(
    input,
    other,
    p,
):
    diff = input - other

    diff_abs = torch.abs(diff).float()

    if p == 0:
        value = torch.count_nonzero(diff)

    elif p == float("inf"):
        value = torch.max(diff_abs)

    elif p == -float("inf"):
        value = torch.min(diff_abs)

    elif p == 1:
        value = torch.sum(diff_abs)

    else:
        value = torch.sum(diff_abs**p)
        value = value ** (1.0 / p)

    return value.to(input.dtype)


def dist(
    input,
    other,
    p=2,
):
    if input.numel() == 0:
        return torch.tensor(
            0.0,
            dtype=input.dtype,
            device=input.device,
        )

    if p == 2:
        return _dist_p2(
            input,
            other,
        )

    return _dist_generic(
        input,
        other,
        p,
    )
