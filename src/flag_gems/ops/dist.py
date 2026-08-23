import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext


@libentry()
@triton.jit
def dist_l2_single_kernel(
    X,
    Y,
    Out,
    N,
    DTYPE_MODE: tl.constexpr,
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

        if DTYPE_MODE == 0:

            x = tl.load(
                X + idx,
                mask=mask,
                other=0.0,
            )

            y = tl.load(
                Y + idx,
                mask=mask,
                other=0.0,
            )

        else:

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
            0.0,
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

    tl.store(
        Partial + pid,
        tl.sum(diff * diff),
    )


@libentry()
@triton.jit
def dist_reduce_kernel(
    Partial,
    Out,
    N,
    BLOCK_SIZE: tl.constexpr,
):

    offsets = tl.arange(
        0,
        BLOCK_SIZE,
    )

    mask = offsets < N

    x = tl.load(
        Partial + offsets,
        mask=mask,
        other=0.0,
    )

    tl.store(
        Out,
        tl.sqrt(tl.sum(x)),
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

    # single kernel path
    if numel <= 65536:

        BLOCK_SIZE = triton.next_power_of_2(min(numel, 4096))

        dtype_mode = 0 if input.dtype == torch.float32 else 1

        dist_l2_single_kernel[(1,)](
            input,
            other,
            out,
            numel,
            DTYPE_MODE=dtype_mode,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    # two-stage reduction
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

        dist_l2_partial_kernel[(partial_size,)](
            input,
            other,
            partial,
            numel,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

        dist_reduce_kernel[(1,)](
            partial,
            out,
            partial_size,
            BLOCK_SIZE=triton.next_power_of_2(partial_size),
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
