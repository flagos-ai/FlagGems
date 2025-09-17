import copy
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry, libtuner

from ..utils import MAX_GRID_SIZE_Y, TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
device = device.name

MAX_C_MLU_CUMSUM = 8192
MAX_C_MLU_SPILT_CUMSUM = 32768
MAX_TILE_N = 256


@triton.jit
def cumsum_blelloch_impl(
    in_block,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_NUM: tl.constexpr,
):
    x_block = tl.reshape(in_block, (BLOCK_M, TILE_NUM, TILE_N, BLOCK_K))
    # Trans TILE_N and apply blelloch in TILE_N dim
    x_block = tl.trans(x_block, 0, 2, 1, 3)
    # Apply blelloch algo
    # Up-Sweep Phase
    step = 1
    while step < TILE_N:
        idx_a = step - 1
        idx_b = idx_a + step
        while idx_b < TILE_N:
            x_block[:, idx_b, :, :] = x_block[:, idx_a, :, :] + x_block[:, idx_b, :, :]
            idx_a += 2 * step
            idx_b += 2 * step
        step *= 2
    # Down-Sweep Phase
    step //= 2
    while step > 0:
        idx_b = TILE_N - 1 - step
        idx_a = idx_b - step
        while idx_a > 0:
            x_block[:, idx_b, :, :] = x_block[:, idx_a, :, :] + x_block[:, idx_b, :, :]
            idx_b -= 2 * step
            idx_a -= 2 * step
        step //= 2
    # Deal the last tile row exclusive sum(Composed by right shift and tl.cumsum)
    # Right shift 1 position for the last tile row
    partial_sum = tl.zeros((BLOCK_M, TILE_NUM, BLOCK_K), dtype=tl.dtype(DTYPE))
    partial_sum[:, 1:, :] = x_block[:, TILE_N - 1, 0 : (TILE_NUM - 1), :]
    partial_sum = tl.cumsum(partial_sum, axis=1)
    # Apply cycle add for all tile data
    x_block += partial_sum[:, None, :, :]
    # Trans TILE_N dim to original pos
    x_block = tl.trans(x_block, 0, 2, 1, 3)
    x_block = tl.reshape(x_block, (BLOCK_M, BLOCK_N, BLOCK_K))
    return x_block


def config_prune(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, TILE_N, num_warps, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            kw["TILE_N"],
            config.num_warps,
            config.num_stages,
        )
        new_config = config
        # When N is less than MAX_C_MLU_CUMSUM, no reduction loops. Unify different BLOCK_N configs.
        if N <= MAX_C_MLU_CUMSUM:
            # change config
            new_config = copy.deepcopy(config)
            BLOCK_N = new_config.kwargs["BLOCK_N"] = triton.next_power_of_2(N)
            num_stages = new_config.num_stages = 1
        else:
            # When N is greater than MAX_C_MLU_CUMSUM, the pruning condition was obtained through experimentation.
            # It may result in not finding the optimal solution.
            if BLOCK_N < 2048:
                continue
            if BLOCK_N >= 2048 and TILE_N < 8:
                continue
            if (
                BLOCK_N < MAX_C_MLU_CUMSUM
                and BLOCK_M < M
                and BLOCK_M <= (MAX_C_MLU_CUMSUM // BLOCK_N * 2)
            ):
                continue
        # BLOCK_M can only be 1 when BLOCK_N is at its maximum
        if BLOCK_N == MAX_C_MLU_CUMSUM and BLOCK_M > 1:
            continue
        # Prune invalid BLOCK_M
        if BLOCK_M > M:
            continue
        # Prune invalid TILE_N
        if TILE_N > BLOCK_N:
            continue
        # The pruning condition was obtained through experimentation. It may result in not finding the optimal solution.
        if BLOCK_N > 128 and TILE_N < 8:
            continue
        key = (BLOCK_M, BLOCK_N, TILE_N, num_warps, num_stages)
        # Only keep one config for the same key
        configs_map.setdefault(key, new_config)
    pruned_configs = []
    for k, v in configs_map.items():
        pruned_configs.append(v)
    return pruned_configs


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {
                "BLOCK_M": m,
                "BLOCK_N": 2**n,
                "TILE_N": 2**t,
            },
            num_stages=s,
            num_warps=1,
        )
        for m in range(1, 20, 3)
        for n in range(7, 13, 1)
        for t in range(0, 7, 1)
        for s in [1, 3]
    ],
    key=[
        "M",
        "N",
        "K",
    ],
    strategy=["log", "log", "log"],
    prune_configs_by={"early_config_prune": config_prune},
)
@triton.heuristics(
    values={
        "TILE_NUM": lambda args: args["BLOCK_N"] // args["TILE_N"]
        if args["BLOCK_N"] % args["TILE_N"] == 0
        and args["BLOCK_N"] // args["TILE_N"] >= 1
        else 1,
        "TILE_N": lambda args: args["BLOCK_N"]
        if args["TILE_NUM"] == 1
        else args["TILE_N"],
    },
)
@triton.jit
def cumsum_blelloch(
    inp,
    out,
    M,
    N,
    K,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_NUM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    kep = tl.full([BLOCK_M, BLOCK_N, 1], float(0), tl.dtype(DTYPE))
    for col_offset in range(0, N, BLOCK_N):
        n_offset = col_offset + tl.arange(0, BLOCK_N)
        # Pointers to the start of the row
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        x_ptrs = inp + offsets
        y_ptrs = out + offsets

        # Load data into NRAM
        in_block = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.dtype(DTYPE))

        x_block = cumsum_blelloch_impl(
            in_block, DTYPE, BLOCK_M, BLOCK_N, 1, TILE_N, TILE_NUM
        )
        # Add last block partial sum to current block
        x_block = tl.reshape(x_block, (BLOCK_M, BLOCK_N))
        kep_tmp = kep[:, BLOCK_N - 1, :]
        x_block += kep_tmp
        kep = x_block[:, :, None]
        # Store result back to global memory
        tl.store(y_ptrs, x_block, mask=mask)


def get_reduction_dim_block_size(N):
    block_size = N // TOTAL_CORE_NUM + ((N % TOTAL_CORE_NUM) != 0)
    if block_size > MAX_C_MLU_SPILT_CUMSUM:
        block_size = MAX_C_MLU_SPILT_CUMSUM
    # In blelloch, block_size = TILE_N * TILE_NUM
    # TILE_N and TILE_NUM should be power of 2, So is it
    return triton.next_power_of_2(block_size)


def config_prune_mid(configs, named_args, **kwargs):
    M = named_args["M"]
    K = named_args["K"]
    BLOCK_N = named_args["BLOCK_N"]
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_K, TILE_N, num_warps, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_K"],
            kw["TILE_N"],
            config.num_warps,
            config.num_stages,
        )
        new_config = config
        # Prune invalid BLOCK_M
        if BLOCK_M > M:
            continue
        # Prune invalid BLOCK_K
        if BLOCK_K > K:
            continue
        if BLOCK_N * BLOCK_K * BLOCK_M > MAX_C_MLU_SPILT_CUMSUM:
            continue
        # Prune invalid TILE_N
        if TILE_N > BLOCK_N:
            continue
        # The pruning condition was obtained through experimentation. It may result in not finding the optimal solution.
        if BLOCK_N > 128 and TILE_N < 8:
            continue
        key = (BLOCK_M, BLOCK_N, BLOCK_K, TILE_N, num_warps, num_stages)
        # Only keep one config for the same key
        configs_map.setdefault(key, new_config)
    pruned_configs = []
    for k, v in configs_map.items():
        pruned_configs.append(v)
    return pruned_configs


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {
                "BLOCK_M": m,
                "BLOCK_K": 2**k,
                "TILE_N": 2**t,
            },
            num_stages=s,
            num_warps=1,
        )
        for m in range(1, 10, 3)
        for k in range(0, 3, 1)
        for t in range(5, int(math.log(MAX_TILE_N, 2) + 1), 1)
        for s in [1, 3]
    ],
    key=[
        "M",
        "N",
        "K",
        "BLOCK_N",
    ],
    strategy=["log", "log", "log", "log"],
    prune_configs_by={"early_config_prune": config_prune_mid},
)
@triton.heuristics(
    values={
        "TILE_NUM": lambda args: args["BLOCK_N"] // args["TILE_N"]
        if args["BLOCK_N"] % args["TILE_N"] == 0
        and args["BLOCK_N"] // args["TILE_N"] >= 1
        else 1,
        "TILE_N": lambda args: args["BLOCK_N"]
        if args["TILE_NUM"] == 1
        else args["TILE_N"],
    },
)
@triton.jit
def cumsum_kernel_mid(
    inp,
    out,
    prefix_sum,
    M,
    N,
    K,
    BLOCK_N: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_NUM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_jobs_n = tl.num_programs(1)
    pid_k = tl.program_id(2)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offset = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = (
        m_offset[:, None, None] * N * K
        + n_offset[
            None,
            :,
            None,
        ]
        * K
        + k_offset[None, None, :]
    )
    mask = (m_offset[:, None, None] < M and n_offset[None, :, None] < N) and k_offset[
        None, None, :
    ] < K
    x_ptrs = inp + offsets
    y_ptrs = out + offsets

    # Load data into NRAM
    in_block = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.dtype(DTYPE))

    x_block = cumsum_blelloch_impl(
        in_block, DTYPE, BLOCK_M, BLOCK_N, BLOCK_K, TILE_N, TILE_NUM
    )
    tl.store(y_ptrs, x_block, mask=mask)
    prefix_sum_offsets = (
        m_offset[:, None] * num_jobs_n * K + pid_n * K + k_offset[None, :]
    )
    prefix_sum_mask = m_offset[:, None] < M and k_offset[None, :] < K
    prefix_sum_ptrs = prefix_sum + prefix_sum_offsets
    tl.store(prefix_sum_ptrs, x_block[:, BLOCK_N - 1, :], prefix_sum_mask)


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {
                "BLOCK_M": m,
                "BLOCK_K": 2**k,
            },
            num_stages=s,
            num_warps=1,
        )
        for m in [1, 3, 6]
        for k in range(0, 3, 1)
        for s in [1, 3]
    ],
    key=[
        "M",
        "N",
        "K",
        "BLOCK_N",
    ],
    strategy=["log", "log", "log", "log"],
)
@triton.jit
def cumsum_kernel_result(
    inp,
    prefix_sum,
    out,
    M,
    N,
    K,
    BLOCK_N: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_jobs_n = tl.num_programs(1)
    pid_k = tl.program_id(2)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offset = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets = (
        m_offset[:, None, None] * N * K
        + n_offset[
            None,
            :,
            None,
        ]
        * K
        + k_offset[None, None, :]
    )
    mask = (m_offset[:, None, None] < M and n_offset[None, :, None] < N) and k_offset[
        None, None, :
    ] < K
    x_ptrs = inp + offsets
    y_ptrs = out + offsets

    # Load data into NRAM
    x_block = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.dtype(DTYPE))

    if pid_n > 0:
        sum_offsets = (
            m_offset[:, None] * num_jobs_n * K + (pid_n - 1) * K + k_offset[None, :]
        )
        sum_mask = m_offset[:, None] < M and k_offset[None, :] < K
        sum_ptrs = prefix_sum + sum_offsets
        sum_block = tl.load(sum_ptrs, mask=sum_mask, other=0.0).to(tl.dtype(DTYPE))
        x_block += sum_block[:, None, :]

    # Store result back to global memory
    tl.store(y_ptrs, x_block, mask=mask)


def cumsum_wrapper(inp, dim=1, dtype=None, out=None):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    inp = inp.contiguous()
    K = inp.numel() // M // N

    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int32
    if out is None:
        out = torch.empty_like(inp, dtype=dtype)

    blelloch_grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )

    dtypestr = "fp32" if torch.is_floating_point(out) else "int32"
    if (M * K < TOTAL_CORE_NUM / 2) and (N > MAX_C_MLU_CUMSUM):
        # result BLOCK_N must be same as mid BLOCK_N
        mid_out = torch.empty_like(inp, dtype=dtype)
        BLOCK_N = get_reduction_dim_block_size(N)
        prefix_sum_inp = torch.empty(
            M, triton.cdiv(N, BLOCK_N), K, dtype=dtype, device=inp.device
        )
        prefix_sum = torch.empty(
            M, triton.cdiv(N, BLOCK_N), K, dtype=dtype, device=inp.device
        )
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, BLOCK_N),
            triton.cdiv(K, meta["BLOCK_K"]),
        )
        with torch_device_fn.device(inp.device):
            cumsum_kernel_mid[grid](
                inp, mid_out, prefix_sum_inp, M, N, K, BLOCK_N, dtypestr
            )
            cumsum_blelloch[blelloch_grid](
                prefix_sum_inp, prefix_sum, M, triton.cdiv(N, BLOCK_N), K, dtypestr
            )
            cumsum_kernel_result[grid](
                mid_out, prefix_sum, out, M, N, K, BLOCK_N, dtypestr
            )
    else:
        with torch_device_fn.device(inp.device):
            cumsum_blelloch[blelloch_grid](inp, out, M, N, K, dtypestr)
    return out


def cumsum(inp, dim=1, *, dtype=None):
    logger.debug("GEMS_CAMBRICON CUMSUM")
    return cumsum_wrapper(inp, dim, dtype)


def cumsum_out(inp, dim=1, *, dtype=None, out):
    logger.debug("GEMS_CAMBRICON CUMSUM_OUT")
    return cumsum_wrapper(inp, dim, dtype, out)


@libentry()
@triton.jit(do_not_specialize=["K"])
def normed_cumsum_kernel(inp, out, K, BLOCK: tl.constexpr):
    row_start = tl.program_id(0) * K
    row_off = tl.arange(0, BLOCK)
    x = tl.load(inp + row_start + row_off, mask=row_off < K, other=0)
    if x.dtype.is_fp16():
        x = x.to(tl.float32)
    y_sum = tl.sum(x, 0)
    y = tl.cumsum(x, 0)
    y = y / y_sum
    tl.store(out + row_start + row_off, y, mask=row_off < K)


@libentry()
@triton.jit(
    do_not_specialize=[
        "r",
        "t",
        "R",
        "K",
        "r_stride",
        "out_r_stride",
    ]
)
def block_cumsum_kernel(
    inp,
    out,
    sums,
    r,
    t,
    R,
    K,
    r_stride,
    k_stride,
    out_r_stride,
    out_k_stride,
    OUTPUT_SUMS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HAS_OUT_LAYOUT: tl.constexpr,
    TILE: tl.constexpr,
):
    # One CTA processes a (r, t*tile) chunk
    # rows = [ grid.y, grid.y + r )
    # cols = [ grid.x * t * tile, (grid.x + 1) * t * tile )
    gridx = tl.program_id(0).to(tl.int64)
    gridy = tl.program_id(1).to(tl.int64)
    n_chunks = tl.num_programs(0)

    for row in range(gridy * r, min((gridy + 1) * r, R)):
        curr_cumsum = tl.zeros((1,), tl.float32)
        row_offset = row * r_stride
        cols = gridx * t * TILE + tl.arange(0, TILE)
        for ti in range(0, t):
            cols_offset = cols * k_stride
            x = tl.load(inp + row_offset + cols_offset, mask=cols < K, other=0)
            if x.dtype.is_fp16() | x.dtype.is_bf16():
                x = x.to(tl.float32)
            tile_sum = tl.sum(x, 0)[None]
            tile_cumsum = tl.cumsum(x, 0) + curr_cumsum
            curr_cumsum += tile_sum
            if HAS_OUT_LAYOUT:
                cols_offset = cols * out_k_stride
                row_offset = row * out_r_stride
            tl.store(out + row_offset + cols_offset, tile_cumsum, mask=cols < K)
            if OUTPUT_SUMS:
                tl.store(sums + row * n_chunks + gridx[None], curr_cumsum)
            cols += TILE
        if NORMALIZE:
            cols = gridx * t * TILE + tl.arange(0, TILE)
            for _ in range(0, t):
                cols_offset = cols * k_stride
                if HAS_OUT_LAYOUT:
                    cols_offset = cols * out_k_stride
                    row_offset = row * out_r_stride
                x = tl.load(out + row_offset + cols_offset, mask=cols < K, other=0)
                if x.dtype.is_fp16() | x.dtype.is_bf16():
                    x = x.to(tl.float32)
                x = x / curr_cumsum
                tl.store(out + row_offset + cols_offset, x, mask=cols < K)
                cols += TILE


@libentry()
@triton.jit(
    do_not_specialize=[
        "r",
        "t",
        "R",
        "K",
        "r_stride",
        "out_r_stride",
    ]
)
def block_update_kernel(
    inp,
    base,
    rscale_ptr,
    out,
    r,
    t,
    R,
    K,
    r_stride,
    k_stride,
    out_r_stride,
    out_k_stride,
    rscale_stride,
    HAS_OUT_LAYOUT: tl.constexpr,
    TILE: tl.constexpr,
):
    # One CTA processes a (r, t*tile) chunk
    # rows = [ grid.y, grid.y + r )
    # cols = [ grid.x * t * tile, (grid.x + 1) * t * tile )
    gridx = tl.program_id(0).to(tl.int64)
    gridy = tl.program_id(1).to(tl.int64)
    n_gridx = tl.num_programs(1)

    base += gridy * n_gridx + gridx
    rscale_ptr += gridy * rscale_stride

    for row in range(gridy, min(gridy + r, R)):
        d = tl.load(base)
        rscale = tl.load(rscale_ptr)
        base += gridx
        rscale_ptr += rscale_stride
        row_offset = row * r_stride
        cols = gridx * t * TILE + tl.arange(0, TILE)
        for _ in range(0, t):
            cols_offset = cols * k_stride
            x = tl.load(inp + row_offset + cols_offset, mask=cols < K, other=0)
            x += d
            x /= rscale
            if HAS_OUT_LAYOUT:
                cols_offset = cols * out_k_stride
                row_offset = row * out_r_stride
            tl.store(out + row_offset + cols_offset, x, mask=cols < K)
            cols += TILE


GRID_Y_LIMIT = MAX_GRID_SIZE_Y


def normed_cumsum(inp, dim=-1):
    logger.debug("GEMS_CAMBRICON NORMED_CUMSUM")
    assert inp.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    dim = dim % inp.ndim
    N = inp.numel()
    K = inp.size(dim)
    # inp = inp.contiguous()
    # First and last dims are easier to handle, but transpose the middle dim to the last
    ranked_dims = sorted(range(inp.ndim), key=lambda i: inp.stride(i), reverse=True)
    is_mid_dim = dim not in (ranked_dims[0], ranked_dims[-1])
    if is_mid_dim:
        inp = inp.transpose(dim, -1).contiguous()
        dim = -1
    out = torch.empty_like(inp)
    with torch_device_fn.device(inp.device.index):
        # Pass one, scan a (batch, n_tiles * TILE) sized block within each cta
        num_sms = TOTAL_CORE_NUM  # torch.cuda.get_device_properties("cuda").multi_processor_count
        TILE = 2048
        # Each row is split into n_chunks of chunks where each chunk is compised of
        # n_tiles of tiles. Different chunks are assigned to different ctas.
        n_rows = N // K
        n_chunks = min(triton.cdiv(num_sms, n_rows), triton.cdiv(K, TILE))
        n_tiles = triton.cdiv(triton.cdiv(K, TILE), n_chunks)
        k_stride = inp.stride(dim)
        r_stride = inp.size(dim) if k_stride == 1 else 1
        if n_rows > GRID_Y_LIMIT:
            batch = triton.cdiv(n_rows, GRID_Y_LIMIT)
            n_batch = triton.cdiv(n_rows, batch)
        else:
            batch = 1
            n_batch = n_rows

        grid = (n_chunks, n_batch)
        if n_chunks == 1:
            block_cumsum_kernel[grid](
                inp,
                out,
                0,
                batch,
                n_tiles,
                n_rows,
                K,
                r_stride,
                k_stride,
                r_stride,
                k_stride,
                OUTPUT_SUMS=False,
                NORMALIZE=True,
                HAS_OUT_LAYOUT=False,
                TILE=TILE,
            )
            return out

        if inp.dtype != torch.float64:
            acc_dtype = torch.float32
        sums = torch.empty((n_rows, n_chunks), dtype=acc_dtype, device=device.name)
        cumsums = torch.empty_like(sums)
        block_cumsum_kernel[grid](
            inp,
            out,
            sums,
            batch,
            n_tiles,
            n_rows,
            K,
            r_stride,
            k_stride,
            r_stride,
            k_stride,
            OUTPUT_SUMS=True,
            NORMALIZE=False,
            HAS_OUT_LAYOUT=False,
            TILE=TILE,
        )
        # Pass two, scan partial cumsums
        block_cumsum_kernel[(1, n_batch)](
            sums,
            cumsums,
            0,
            batch,
            1,
            n_rows,
            n_chunks,
            n_chunks,
            1,
            n_chunks,
            1,
            OUTPUT_SUMS=False,
            NORMALIZE=False,
            HAS_OUT_LAYOUT=True,
            TILE=TILE,
        )
        # print(sums)
        rscale = cumsums[..., -1]
        block_update_kernel[grid](
            out,
            cumsums - sums,
            rscale,
            out,
            batch,
            n_tiles,
            n_rows,
            K,
            r_stride,
            k_stride,
            r_stride,
            k_stride,
            n_chunks,
            HAS_OUT_LAYOUT=False,
            TILE=TILE,
        )
        return out
