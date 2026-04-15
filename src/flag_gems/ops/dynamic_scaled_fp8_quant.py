from typing import Optional

import torch
import triton
import triton.language as tl

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
FP8_MIN = float(torch.finfo(FP8_DTYPE).min)
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


def _get_fp8_quant_2d_config(
    m: int,
    n: int,
    *,
    single_launch: bool,
) -> dict[str, int]:
    if n <= 512:
        block_n = 512
        num_warps = 4
        num_stages = 1
    elif n <= 1024:
        block_n = 1024
        num_warps = 4 if single_launch or m <= 4 else 8
        num_stages = 2
    elif n <= 2048:
        block_n = 2048
        num_warps = 8
        num_stages = 2
    else:
        block_n = 4096
        num_warps = 8
        num_stages = 2

    return {
        "BLOCK_N": block_n,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _get_fp8_quant_single_cta_config(numel: int) -> dict[str, int]:
    if numel <= 1024:
        block_size = 1024
        num_warps = 4
    elif numel <= 2048:
        block_size = 2048
        num_warps = 8
    elif numel <= 4096:
        block_size = 4096
        num_warps = 16
    elif numel <= 8192:
        block_size = 8192
        num_warps = 16
    else:
        block_size = 16384
        num_warps = 16

    return {
        "BLOCK_SIZE": block_size,
        "num_warps": num_warps,
        "num_stages": 2,
    }


@triton.jit
def global_absmax_atomic_kernel(
    inp_ptr,
    absmax_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_max(absmax_ptr, tl.max(tl.abs(x), axis=0))


@triton.jit
def dynamic_scaled_fp8_quant_single_launch_kernel(
    out_ptr,
    inp_ptr,
    scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_om,
    stride_on,
    stride_im,
    stride_in,
    fp8_min,
    fp8_max,
    min_scale,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid != 0:
        return

    global_absmax = 0.0
    for pid_m in tl.static_range(0, M):
        row_base = pid_m * stride_im
        for off in tl.static_range(0, N, BLOCK_N):
            offs_n = off + tl.arange(0, BLOCK_N)
            mask = offs_n < N
            x = tl.load(
                inp_ptr + row_base + offs_n * stride_in,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            global_absmax = tl.maximum(global_absmax, tl.max(tl.abs(x), axis=0))

    scale = tl.maximum(global_absmax / fp8_max, min_scale)
    tl.store(scale_ptr, scale)
    inv_scale = 1.0 / scale

    for pid_m in tl.static_range(0, M):
        row_in_base = pid_m * stride_im
        row_out_base = pid_m * stride_om
        for off in tl.static_range(0, N, BLOCK_N):
            offs_n = off + tl.arange(0, BLOCK_N)
            mask = offs_n < N
            x = tl.load(
                inp_ptr + row_in_base + offs_n * stride_in,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            q = tl.clamp(x * inv_scale, fp8_min, fp8_max)
            tl.store(
                out_ptr + row_out_base + offs_n * stride_on,
                q.to(out_ptr.type.element_ty),
                mask=mask,
            )


@triton.jit
def dynamic_scaled_fp8_quant_large_m_kernel(
    out_ptr,
    inp_ptr,
    scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_om,
    stride_on,
    stride_im,
    stride_in,
    fp8_min,
    fp8_max,
    min_scale,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    row_in_base = pid_m * stride_im
    row_out_base = pid_m * stride_om
    scale = tl.load(scale_ptr).to(tl.float32)
    inv_scale = 1.0 / tl.maximum(scale, min_scale)

    for off in tl.static_range(0, N, BLOCK_N):
        offs_n = off + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        x = tl.load(
            inp_ptr + row_in_base + offs_n * stride_in,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        q = tl.clamp(x * inv_scale, fp8_min, fp8_max)
        tl.store(
            out_ptr + row_out_base + offs_n * stride_on,
            q.to(out_ptr.type.element_ty),
            mask=mask,
        )


def _dynamic_scaled_fp8_quant_small_m(
    output: torch.Tensor,
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_out = torch.empty((1,), device=input.device, dtype=torch.float32)
    config = _get_fp8_quant_single_cta_config(input.shape[-1])
    dynamic_scaled_fp8_quant_single_launch_kernel[(1,)](
        output,
        input,
        scale_out,
        input.shape[0],
        input.shape[1],
        output.stride(0),
        output.stride(1),
        input.stride(0),
        input.stride(1),
        fp8_min=FP8_MIN,
        fp8_max=FP8_MAX,
        min_scale=FP8_MIN_SCALE,
        BLOCK_N=config["BLOCK_SIZE"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return output, scale_out


def _dynamic_scaled_fp8_quant_large_m(
    output: torch.Tensor,
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = input.shape
    config = _get_fp8_quant_2d_config(m, n, single_launch=False)
    absmax = torch.zeros((1,), device=input.device, dtype=torch.float32)
    block = 512

    global_absmax_atomic_kernel[(triton.cdiv(input.numel(), block),)](
        input,
        absmax,
        input.numel(),
        BLOCK_SIZE=block,
        num_warps=4,
        num_stages=1,
    )

    scale_out = (absmax / FP8_MAX).clamp_(min=FP8_MIN_SCALE)
    dynamic_scaled_fp8_quant_large_m_kernel[(m,)](
        output,
        input,
        scale_out,
        m,
        n,
        output.stride(0),
        output.stride(1),
        input.stride(0),
        input.stride(1),
        fp8_min=FP8_MIN,
        fp8_max=FP8_MAX,
        min_scale=FP8_MIN_SCALE,
        BLOCK_N=config["BLOCK_N"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return output, scale_out


def dynamic_scaled_fp8_quant(
    input: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert input.ndim == 2 and input.stride(-1) == 1

    if output is None:
        output = torch.empty_like(input, dtype=FP8_DTYPE)
    else:
        assert output.shape == input.shape and output.dtype == FP8_DTYPE

    use_single_launch_kernel = input.shape[0] <= 8 and input.numel() <= 65536
    if use_single_launch_kernel:
        return _dynamic_scaled_fp8_quant_small_m(output, input)
    return _dynamic_scaled_fp8_quant_large_m(output, input)
