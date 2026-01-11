
import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

@libentry()
@triton.autotune(
    configs=[
        triton.Config({"TILE_M": 128, "TILE_N": 128, "TILE_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"TILE_M": 128, "TILE_N": 128, "TILE_K": 64}, num_stages=4, num_warps=8),
        
        triton.Config({"TILE_M": 128, "TILE_N": 256, "TILE_K": 64}, num_stages=2, num_warps=8),
        triton.Config({"TILE_M": 128, "TILE_N": 256, "TILE_K": 128}, num_stages=2, num_warps=8),
        
        triton.Config({"TILE_M": 64, "TILE_N": 64, "TILE_K": 64}, num_stages=4, num_warps=4),
    ],
    key=["_M_NPO2", "N", "K"],
)
@triton.jit
def _int8_gemm_unmasked_kernel(
    c_ptr,
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    _M_NPO2,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACC_DTYPE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    IS_PER_TOKEN_A: tl.constexpr,
    IS_PER_TOKEN_B: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    pid = tl.program_id(axis=0)
    

    num_pid_m = tl.cdiv(M, TILE_M)
    num_pid_n = tl.cdiv(N, TILE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Block Pointers
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((TILE_M, TILE_N), dtype=ACC_DTYPE)
    
 
    a_step_ptr = TILE_K * stride_ak
    b_step_ptr = TILE_K * stride_bk


    for k in range(0, K, TILE_K):
        mask_k = (k + offs_k) < K
        
        load_mask_a = mask_m[:, None] & mask_k[None, :]
        load_mask_b = mask_k[:, None] & mask_n[None, :]
        
        # Load with mask
        a = tl.load(a_ptrs, mask=load_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=load_mask_b, other=0.0)
        
        # Matrix multiplication
        acc = tl.dot(a, b, acc, out_dtype=ACC_DTYPE)

        # Advance pointers
        a_ptrs += a_step_ptr
        b_ptrs += b_step_ptr

    acc = acc.to(tl.float32)

    # Scale A
    if IS_PER_TOKEN_A:
        # offs_m is 1D (TILE_M,), need 1D mask
        scale_a = tl.load(a_scale_ptr + offs_m, mask=mask_m, other=1.0)
        acc = acc * scale_a[:, None]
    else:
        scale_a = tl.load(a_scale_ptr)
        acc = acc * scale_a

    # Scale B
    if IS_PER_TOKEN_B:
        # offs_n is 1D (TILE_N,), need 1D mask
        scale_b = tl.load(b_scale_ptr + offs_n, mask=mask_n, other=1.0)
        acc = acc * scale_b[None, :]
    else:
        scale_b = tl.load(b_scale_ptr)
        acc = acc * scale_b

    # Bias
    if bias_ptr is not None:
        # offs_n is 1D (TILE_N,), need 1D mask
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]

    # Store
    c = acc.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=store_mask)



def int8_gemm(a, w, a_scale, w_scale, bias=None, out_dtype=torch.float16):
    if a.dtype != torch.int8 or w.dtype != torch.int8:
        raise TypeError(f"Expected int8 inputs, got a={a.dtype}, w={w.dtype}")
    if a.dim() != 2 or w.dim() != 2:
        raise ValueError("Expected 2D tensors: a (M,K), w (K,N)")
    if a.shape[1] != w.shape[0]:
        raise ValueError("incompatible dimensions: a.shape[1] must equal w.shape[0]")
    if out_dtype not in (torch.float16, torch.float32):
        raise TypeError(f"out_dtype must be torch.float16 or torch.float32, got {out_dtype}")

    if a.device != w.device:
        raise ValueError("a and w must be on the same device")
    device = a.device

    # handle non-contiguous inputs if necessary (match mm.py style)
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if w.stride(0) > 1 and w.stride(1) > 1:
        w = w.contiguous()

    M, K = a.shape
    _, N = w.shape
    
    if torch.is_tensor(a_scale):
        a_scale_t = a_scale.to(device=device, dtype=torch.float32).contiguous()
    else:
        a_scale_t = torch.tensor([float(a_scale)], device=device, dtype=torch.float32)
    
    if torch.is_tensor(w_scale):
        if w_scale.numel() != 1 and (w_scale.dim() != 1 or w_scale.shape[0] != N):
             raise ValueError(f"w_scale must be scalar or shape (N,), got {tuple(w_scale.shape)}")
        w_scale_t = w_scale.to(device=device, dtype=torch.float32).contiguous()
    else:
        w_scale_t = torch.tensor([float(w_scale)], device=device, dtype=torch.float32)

    if bias is not None:
        if not torch.is_tensor(bias):
             raise TypeError("bias must be a torch.Tensor or None")
        if bias.dim() != 1 or bias.shape[0] != N:
             raise ValueError(f"bias must have shape (N,), got {tuple(bias.shape)}")
        bias = bias.to(device).contiguous()

    c = torch.empty((M, N), device=device, dtype=out_dtype)

    grid = lambda META: (
        triton.cdiv(M, META["TILE_M"]) * triton.cdiv(N, META["TILE_N"]),
    )
    
    IS_PER_TOKEN_A = a_scale_t.numel() > 1
    IS_PER_TOKEN_B = w_scale_t.numel() > 1

    _int8_gemm_unmasked_kernel[grid](
        c, a, w,
        a_scale_t, w_scale_t, bias,
        M, N, K,
        triton.next_power_of_2(M),
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        c.stride(0), c.stride(1),
        ACC_DTYPE=tl.int32,
        IS_PER_TOKEN_A=IS_PER_TOKEN_A,
        IS_PER_TOKEN_B=IS_PER_TOKEN_B,
    )
    return c
