import torch
import triton
import triton.language as tl


@triton.jit
def rwkv_ka_fusion_kernel(
    k_ptr,
    kk_ptr,
    a_ptr,
    ka_ptr,
    o_k_ptr,
    o_kk_ptr,
    o_kka_ptr,
    T,
    C,
    H,
    N,
    N_size: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    k_start = pid * block_size

    for i in range(0, H):
        offs = k_start + i * N + tl.arange(0, N_size)
        k = tl.load(k_ptr + offs, mask=offs < T * C, other=0.0)
        a = tl.load(a_ptr + offs, mask=offs < T * C, other=0.0)

        c_offs = i * N + tl.arange(0, N_size)
        ka = tl.load(ka_ptr + c_offs, mask=c_offs < C, other=0.0)
        kk = tl.load(kk_ptr + c_offs, mask=c_offs < C, other=0.0)

        kt = k * kk
        kt2 = kt * kt
        norm_kt2 = tl.sum(kt2.to(tl.float32))
        norm_kt = tl.sqrt(norm_kt2 + 1e-12)
        okk = kt / norm_kt
        tl.store(o_kk_ptr + offs, okk, mask=offs < T * C)

        ok = k * (1 + (a.to(tl.float32) - 1) * ka)
        okka = okk * a
        tl.store(o_k_ptr + offs, ok, mask=offs < T * C)
        tl.store(o_kka_ptr + offs, okka, mask=offs < T * C)


def rwkv_ka_fusion(
    k: torch.Tensor, kk: torch.Tensor, a: torch.Tensor, ka: torch.Tensor, H: int, N: int
):
    if k.dim() == 1:
        T = 1
        C = k.shape[0]
    else:
        T, C = k.shape

    o_k = torch.empty_like(k)
    o_kk = torch.empty_like(k)
    o_kka = torch.empty_like(k)

    BLOCK_SIZE = 1 * C
    grid = lambda meta: (triton.cdiv(T * C, BLOCK_SIZE),)
    N_size = triton.next_power_of_2(N)
    rwkv_ka_fusion_kernel[grid](
        k, kk, a, ka, o_k, o_kk, o_kka, T, C, H, N, N_size, BLOCK_SIZE
    )

    return o_k, o_kk, o_kka
