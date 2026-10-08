import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.jit
def _copy_1d_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    offs64 = offs.to(tl.int64)
    x = tl.load(in_ptr + offs64, mask=mask)
    tl.store(out_ptr + offs64, x, mask=mask)


@triton.jit
def _t_copy_col_kernel(in_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
    # Transpose copy of a contiguous (M, N) input into a contiguous (N, M)
    # output. One program handles output row `c` (= input column c): it reads
    # that input column's M elements with stride N (DISCRETE read) and writes
    # them into output row c CONTIGUOUSLY (block DMA).
    #
    # A transpose has one inherently strided side and XPU-Triton cannot do an
    # in-SRAM register transpose (`tl.trans` fails to compile here), so exactly
    # one side is discrete. Measured on XPU a DISCRETE READ is ~1.5x cheaper
    # than a discrete WRITE (reads pipeline/prefetch, writes must commit), so
    # we put the strided access on the READ and keep the WRITE contiguous:
    # 4096^2 fp16 col-kernel ~9.2ms vs row-kernel ~14ms. See
    # solution/t_copy_perf_fix.md.
    c = tl.program_id(0)  # output row index / input column index [0, N)
    pm = tl.program_id(1)
    c64 = c.to(tl.int64)
    out_base = c64 * M
    r = pm * BLOCK + tl.arange(0, BLOCK)
    mask = r < M
    r64 = r.to(tl.int64)
    x = tl.load(in_ptr + r64 * N + c64, mask=mask)  # discrete read (stride N)
    tl.store(out_ptr + out_base + r64, x, mask=mask)  # contiguous write


# SDNN tile-transpose kernel (TritonSDNN pipeline): the pipeline DMAs the
# (MT, NT) tile GM -> uni_sram, `tl.trans` performs the on-chip transpose,
# and the pipeline DMAs the (NT, MT) tile uni_sram -> GM. Both GM sides stay
# row-contiguous 2D block transfers; no per-element gather. This removes the
# per-element gather that caps the generic (col) kernel at ~10G load/s:
# measured ~40-45us vs 1.78ms on f16 4096x4096.
#
# NOTE: only usable with the TritonSDNN pipeline (is_sdnn=True); the generic
# pipeline cannot lower tl.trans for this arch.
@triton.jit(do_not_specialize=["In", "Out", "M", "N"])
def _t_copy_tile_kernel(In, Out, M, N, MT: tl.constexpr, NT: tl.constexpr):
    pid = tl.program_id(0)
    ntx = N // NT
    r0 = (pid // ntx) * MT
    c0 = (pid % ntx) * NT
    rm = tl.arange(0, MT)
    rn = tl.arange(0, NT)
    x = tl.load(In + (r0 + rm)[:, None] * N + (c0 + rn)[None, :])
    y = tl.trans(x)
    tl.store(Out + (c0 + rn)[:, None] * M + (r0 + rm)[None, :], y)


def _pick_t_copy_cfg(M: int, N: int, dtype: torch.dtype):
    # Configs measured on P800 (KL3); 64x64 is small enough that any tile
    # covers the whole tensor and only launch overhead matters.
    if dtype is torch.float32:
        if M % 128 == 0 and N % 512 == 0:
            return 128, 512, 2, 4
    else:
        if M % 256 == 0 and N % 256 == 0:
            return 256, 256, 3, 4
    if M % 64 == 0 and N % 64 == 0:
        return 64, 64, 2, 4
    return None


def _launch_t_copy_tile(inp, out, M, N, MT, NT, num_stages, num_warps):
    if inp.dtype is torch.bfloat16:
        # The pipeline stores bf16 WIDENED (32-bit) in uni_sram, doubling
        # on-chip traffic; an f16 bitcast view keeps the transpose a pure
        # 2-byte bit permutation (tl.trans is layout only, no arithmetic).
        src = inp.view(torch.float16)
        dst = out.view(torch.float16)
    else:
        src, dst = inp, out
    grid = ((M // MT) * (N // NT),)
    _t_copy_tile_kernel[grid](
        src, dst, M, N, MT, NT, is_sdnn=True, num_stages=num_stages, num_warps=num_warps
    )


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _launch_t_copy_kernel(inp: torch.Tensor, out: torch.Tensor):
    if inp.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"t_copy kernels require {flag_gems.device} tensors")
    assert inp.dtype == out.dtype, "dtype mismatch between input and output"

    dim = inp.dim()
    if dim <= 1:
        # 0-D or 1-D: pure contiguous copy.
        n = inp.numel()
        assert out.numel() == n, "Output size mismatch for t_copy"
        if n == 0:
            return
        src = inp if inp.is_contiguous() else inp.contiguous()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _copy_1d_kernel[grid](src.view(-1), out.view(-1), n, BLOCK_SIZE=1024)
    elif dim == 2:
        M, N = inp.shape
        assert (
            out.dim() == 2 and out.shape[0] == N and out.shape[1] == M
        ), "Output shape must be (input.size(1), input.size(0)) for t_copy"
        if inp.numel() == 0:
            return
        # Read side is strided (transpose); keep the input contiguous so the
        # stride is a clean N (materialize a copy if the input view is itself
        # strided, which is rare -- t() of a normal tensor is contiguous).
        src = inp if inp.is_contiguous() else inp.contiguous()
        # SDNN fast path: on-chip transpose (tl.trans) with the pipeline's
        # DMA on both GM sides. Requires 2D contiguous, 64B aligned pointers
        # and tile dims that are multiples of 64.
        if (
            inp.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and out.is_contiguous()
            and M % 64 == 0
            and N % 64 == 0
            and src.data_ptr() % 64 == 0
            and out.data_ptr() % 64 == 0
        ):
            cfg = _pick_t_copy_cfg(M, N, inp.dtype)
            if cfg is not None:
                MT, NT, num_stages, num_warps = cfg
                _launch_t_copy_tile(src, out, M, N, MT, NT, num_stages, num_warps)
                return
        block = min(_next_pow2(M), 4096)
        grid = (N, triton.cdiv(M, block))
        _t_copy_col_kernel[grid](src, out, M, N, BLOCK=block)
    else:
        raise RuntimeError("t_copy expects a tensor with <= 2 dims")


def t_copy_out(
    input: torch.Tensor,
    out: torch.Tensor,
    memory_format: torch.memory_format | None = None,
):
    logger.debug("GEMS_KUNLUNXIN T_COPY_OUT")
    _launch_t_copy_kernel(input, out)
    return out


def t_copy(input: torch.Tensor, memory_format: torch.memory_format | None = None):
    logger.debug("GEMS_KUNLUNXIN T_COPY")
    dim = input.dim()
    if dim == 0:
        out = torch.empty((), dtype=input.dtype, device=input.device)
    elif dim == 1:
        out = torch.empty_like(input, memory_format=torch.contiguous_format)
    elif dim == 2:
        M, N = input.shape
        out = torch.empty(
            (N, M),
            dtype=input.dtype,
            device=input.device,
            memory_format=torch.contiguous_format,
        )
    else:
        raise RuntimeError("t_copy expects a tensor with <= 2 dims")
    _launch_t_copy_kernel(input, out)
    return out
