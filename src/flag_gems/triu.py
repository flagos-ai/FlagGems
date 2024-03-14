import torch
import triton
import triton.language as tl
from .libentry import libentry

def cfggen():
    warps = [1, 2, 4, 8, 16, 32]
    configs = [
        triton.Config({"M_BLOCK_SIZE": 1, "N_BLOCK_SIZE": 2048}, num_warps=w)
        for w in warps
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N", "diagonal"])
@triton.jit
def triu_kernel(
    X,
    Y,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)[:, None]
    m_mask = row < M
    X += row * N
    Y += row * N

    for n_offset in range(0, N, N_BLOCK_SIZE):
        cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
        n_mask = cols < N
        mask = m_mask and n_mask
        
        x = tl.load(X+cols, mask, other=0.0)
        y = tl.where(row + diagonal <= cols, x, 0.0)
        tl.store(Y + cols, y, mask=mask)
    


def triu(A, diagonal=0, *, out=None):
    if __debug__:
        print("FLAG TRIU")
    if out == None:
        O = torch.empty_like(A)
    else:
        O = out
    assert len(A.shape) == 2, "Input tensor should be with 2 dim"
    M, N = A.shape

    grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
    triu_kernel[grid](A, O, M, N, diagonal)
    return O
