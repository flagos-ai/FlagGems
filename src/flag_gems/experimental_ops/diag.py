import torch
import triton
import triton.language as tl


@triton.jit
def _diag_extract_kernel(
    a_ptr, out_ptr, i0, j0, L, stride_row, stride_col, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < L
    a_idx = (i0 + offs) * stride_row + (j0 + offs) * stride_col
    vals = tl.load(a_ptr + a_idx, mask=mask)
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def _diag_write_kernel(
    v_ptr, out_ptr, i0, j0, N, stride_row, stride_col, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    out_idx = (i0 + offs) * stride_row + (j0 + offs) * stride_col
    vals = tl.load(v_ptr + offs, mask=mask)
    tl.store(out_ptr + out_idx, vals, mask=mask)


def diag(*args, **kwargs):
    # Parse input tensor and diagonal
    if len(args) < 1 or not isinstance(args[0], torch.Tensor):
        raise TypeError("diag expects a torch.Tensor as the first positional argument")
    input = args[0]
    diagonal = 0
    if len(args) >= 2 and isinstance(args[1], int):
        diagonal = args[1]
    elif "diagonal" in kwargs and isinstance(kwargs["diagonal"], int):
        diagonal = kwargs["diagonal"]

    if input.dim() == 1:
        N = input.numel()
        k = int(diagonal)
        i0 = max(0, -k)
        j0 = max(0, k)
        size = N + abs(k)
        out = torch.zeros((size, size), dtype=input.dtype, device=input.device)
        if N > 0:
            stride_row, stride_col = out.stride()
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
            _diag_write_kernel[grid](
                input, out, i0, j0, N, stride_row, stride_col, BLOCK_SIZE=1024
            )
        return out
    elif input.dim() == 2:
        M, Nv = input.shape
        k = int(diagonal)
        i0 = max(0, -k)
        j0 = max(0, k)
        L = min(M - i0, Nv - j0)
        L = max(L, 0)
        out = torch.empty((L,), dtype=input.dtype, device=input.device)
        if L > 0:
            stride_row, stride_col = input.stride()
            grid = lambda meta: (triton.cdiv(L, meta["BLOCK_SIZE"]),)
            _diag_extract_kernel[grid](
                input, out, i0, j0, L, stride_row, stride_col, BLOCK_SIZE=1024
            )
        return out
    else:
        raise RuntimeError("diag expects a 1D or 2D tensor")


def diag_out(*args, **kwargs):
    # Supports signatures:
    # - diag_out(input, diagonal, out)
    # - diag_out(out, input, diagonal)
    # - diag_out(input, diagonal, out=...)
    # - diag_out(input, out=..., diagonal=...)
    input = None
    out = None
    diagonal = 0

    # Extract out from kwargs if provided
    if "out" in kwargs and isinstance(kwargs["out"], torch.Tensor):
        out = kwargs["out"]

    # Try positional interpretations
    if input is None and len(args) >= 1 and isinstance(args[0], torch.Tensor):
        # Could be (input, diagonal, out) or (out, input, diagonal)
        if (
            out is None
            and len(args) >= 3
            and isinstance(args[2], torch.Tensor)
            and isinstance(args[1], int)
        ):
            input = args[0]
            diagonal = int(args[1])
            out = args[2]
        elif (
            out is None
            and len(args) >= 3
            and isinstance(args[0], torch.Tensor)
            and isinstance(args[1], torch.Tensor)
            and isinstance(args[2], int)
        ):
            out = args[0]
            input = args[1]
            diagonal = int(args[2])
        else:
            # Fallback: treat first tensor as input
            input = args[0]
            if len(args) >= 2 and isinstance(args[1], int):
                diagonal = int(args[1])

    # Override diagonal from kwargs if provided
    if "diagonal" in kwargs and isinstance(kwargs["diagonal"], int):
        diagonal = int(kwargs["diagonal"])

    if input is None or out is None:
        raise TypeError("diag_out expects input tensor, diagonal, and out tensor")

    if input.dim() == 1:
        N = input.numel()
        k = int(diagonal)
        i0 = max(0, -k)
        j0 = max(0, k)
        size = N + abs(k)

        if out.dim() != 2 or out.shape[0] != size or out.shape[1] != size:
            raise RuntimeError(
                f"diag_out: expected out shape ({size}, {size}), got {tuple(out.shape)}"
            )
        if out.dtype != input.dtype or out.device != input.device:
            raise RuntimeError("diag_out: out dtype/device must match input")

        # Zero-fill out and write diagonal
        if out.numel() > 0:
            out.zero_()
        if N > 0:
            stride_row, stride_col = out.stride()
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
            _diag_write_kernel[grid](
                input, out, i0, j0, N, stride_row, stride_col, BLOCK_SIZE=1024
            )
        return out
    elif input.dim() == 2:
        M, Nv = input.shape
        k = int(diagonal)
        i0 = max(0, -k)
        j0 = max(0, k)
        L = min(M - i0, Nv - j0)
        L = max(L, 0)

        if out.dim() != 1 or out.numel() != L:
            raise RuntimeError(
                f"diag_out: expected out shape ({L},), got {tuple(out.shape)}"
            )
        if out.dtype != input.dtype or out.device != input.device:
            raise RuntimeError("diag_out: out dtype/device must match input")

        if L > 0:
            stride_row, stride_col = input.stride()
            grid = lambda meta: (triton.cdiv(L, meta["BLOCK_SIZE"]),)
            _diag_extract_kernel[grid](
                input, out, i0, j0, L, stride_row, stride_col, BLOCK_SIZE=1024
            )
        return out
    else:
        raise RuntimeError("diag_out expects a 1D or 2D input tensor")
