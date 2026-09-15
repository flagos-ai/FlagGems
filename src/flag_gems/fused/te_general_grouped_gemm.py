# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TransformerEngine-compatible general_grouped_gemm implementation using Triton.

This module provides a Triton-based implementation of grouped GEMM operations
that is API-compatible with TransformerEngine's tex.te_general_grouped_gemm.

Supports:
- Multiple GEMM layouts: TN, NN, NT
- Bias addition with optional GELU activation
- Single output mode (concatenated output) and discrete output mode
- Gradient computation mode
- Accumulation into existing output tensors
"""

import logging
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, libtuner, tl_extra_shim
from flag_gems.utils.device_info import get_device_capability, get_sm_count

# Import tanh from tl_extra_shim for Triton version compatibility
tanh = tl_extra_shim.tanh

logger = logging.getLogger(__name__)


def supports_tma():
    """Check if the device supports TMA (Tensor Memory Access)."""
    return get_device_capability()[0] >= 9


# Check for TMA support
if hasattr(tl, "make_tensor_descriptor"):
    _support_device_tensor_descriptor = True
    make_tensor_descriptor_fn = tl.make_tensor_descriptor
else:
    _support_device_tensor_descriptor = False
    make_tensor_descriptor_fn = None


def get_autotune_config():
    """Get autotuning configurations for GEMM kernel."""
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
    ]


@triton.jit
def gelu_tanh_approx(x):
    """GELU activation function (tanh approximation)."""
    return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


@libentry()
@libtuner(configs=get_autotune_config(), key=["M", "N", "K"])
@triton.jit
def single_gemm_kernel(
    # Pointers
    A,
    B,
    C,
    bias_ptr,
    pre_gelu_ptr,
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    # Flags
    HAS_BIAS: tl.constexpr,
    HAS_GELU: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    TRANSA: tl.constexpr,
    TRANSB: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Single GEMM kernel with configurable layout.

    Computes C = A @ B (with optional transpose, bias, GELU, accumulate).
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize pointers for A and B
    if TRANSA:
        # A is stored as (K, M)
        a_ptrs = A + (offs_k[:, None] * stride_am + offs_am[None, :] * stride_ak)
    else:
        # A is stored as (M, K)
        a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    if TRANSB:
        # B is stored as (N, K)
        b_ptrs = B + (offs_bn[:, None] * stride_bk + offs_k[None, :] * stride_bn)
    else:
        # B is stored as (K, N)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Main loop
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if TRANSA:
            a = tl.load(a_ptrs, mask=(offs_k[:, None] < K) & (offs_am[None, :] < M), other=0.0)
            a = tl.trans(a)
        else:
            a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        if TRANSB:
            b = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & (offs_k[None, :] < K), other=0.0)
            b = tl.trans(b)
        else:
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0.0)

        accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)

        # Advance pointers
        offs_k += BLOCK_K
        if TRANSA:
            a_ptrs += BLOCK_K * stride_am
        else:
            a_ptrs += BLOCK_K * stride_ak
        if TRANSB:
            b_ptrs += BLOCK_K * stride_bn
        else:
            b_ptrs += BLOCK_K * stride_bk

    # Add bias
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn * stride_bias, mask=offs_bn < N, other=0.0)
        accumulator = accumulator + bias[None, :]

    # Store pre-GELU and apply GELU
    if HAS_GELU:
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        pre_gelu_ptrs = pre_gelu_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        pre_gelu_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(pre_gelu_ptrs, accumulator.to(tl.float16), mask=pre_gelu_mask)
        accumulator = gelu_tanh_approx(accumulator)

    # Handle accumulation
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if ACCUMULATE:
        c_old = tl.load(c_ptrs, mask=c_mask, other=0.0)
        accumulator = accumulator + c_old.to(tl.float32)

    # Store output
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)


def _compute_gemm_dimensions(
    A: torch.Tensor,
    B: torch.Tensor,
    transa: bool,
    transb: bool,
) -> Tuple[int, int, int]:
    """
    Compute GEMM dimensions based on layout.

    Layout convention (same as TransformerEngine):
    - TN: A stored as (K, M), B stored as (K, N) -> A^T @ B = (M, K) @ (K, N) = (M, N)
    - NN: A stored as (M, K), B stored as (K, N) -> A @ B = (M, K) @ (K, N) = (M, N)
    - NT: A stored as (M, K), B stored as (N, K) -> A @ B^T = (M, K) @ (K, N) = (M, N)

    Returns:
        (M, N, K)
    """
    if transa:
        K, M = A.shape
    else:
        M, K = A.shape

    if transb:
        N, K_b = B.shape
    else:
        K_b, N = B.shape

    assert K == K_b, f"K dimension mismatch: A has K={K}, B has K={K_b}"
    return M, N, K


def _launch_single_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    transa: bool,
    transb: bool,
    bias: Optional[torch.Tensor] = None,
    pre_gelu_out: Optional[torch.Tensor] = None,
    accumulate: bool = False,
):
    """Launch single GEMM kernel."""
    M, N, K = _compute_gemm_dimensions(A, B, transa, transb)

    if M == 0 or N == 0 or K == 0:
        if not accumulate:
            out.zero_()
        return

    # Compute strides based on layout
    if transa:
        stride_am, stride_ak = A.stride(0), A.stride(1)
    else:
        stride_am, stride_ak = A.stride(0), A.stride(1)

    if transb:
        stride_bk, stride_bn = B.stride(0), B.stride(1)
    else:
        stride_bk, stride_bn = B.stride(0), B.stride(1)

    stride_cm, stride_cn = out.stride(0), out.stride(1)

    # Handle optional tensors - use dummy tensors if not provided
    if bias is not None:
        bias_tensor = bias
        stride_bias = bias.stride(0)
    else:
        # Create a dummy scalar tensor
        bias_tensor = torch.zeros(1, dtype=out.dtype, device=out.device)
        stride_bias = 0

    if pre_gelu_out is not None:
        pre_gelu_tensor = pre_gelu_out
    else:
        pre_gelu_tensor = torch.zeros(1, dtype=out.dtype, device=out.device)

    # Grid
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    # Launch kernel
    single_gemm_kernel[grid](
        A,
        B,
        out,
        bias_tensor,
        pre_gelu_tensor,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias,
        HAS_BIAS=bias is not None,
        HAS_GELU=pre_gelu_out is not None,
        ACCUMULATE=accumulate,
        TRANSA=transa,
        TRANSB=transb,
    )


def te_general_grouped_gemm(
    A: List[torch.Tensor],
    transa: bool,
    B: List[torch.Tensor],
    transb: bool,
    D: Optional[List[torch.Tensor]],
    D_type: torch.dtype,
    m_splits: List[int],
    bias: List[torch.Tensor],
    bias_type: torch.dtype,
    single_output: bool,
    pre_gelu_out: List[torch.Tensor],
    grad: bool,
    workspace: List[torch.Tensor],
    workspaceSize: int,
    accumulate: bool,
    use_split_accumulator: bool,
    math_sm_count: int,
) -> Optional[List[torch.Tensor]]:
    """
    TransformerEngine-compatible te_general_grouped_gemm implementation using Triton.

    This API matches tex.te_general_grouped_gemm from TransformerEngine C++ extension.

    Args:
        A: List of input A tensors
        transa: Whether to transpose A
        B: List of input B tensors
        transb: Whether to transpose B
        D: Optional list of output tensors
        D_type: Output data type
        m_splits: M dimension splits for single_output mode
        bias: List of bias tensors (or grad_bias if grad=True)
        bias_type: Bias data type
        single_output: Whether to use single concatenated output
        pre_gelu_out: List of pre-GELU output tensors
        grad: Whether this is gradient computation
        workspace: List of workspace tensors (ignored, for API compatibility)
        workspaceSize: Workspace size (ignored, for API compatibility)
        accumulate: Whether to accumulate into output
        use_split_accumulator: Whether to use split accumulator (ignored)
        math_sm_count: Number of SMs to use (ignored, for API compatibility)

    Returns:
        List of bias tensors (may contain gradients if grad=True)
    """
    num_gemms = len(A)
    if num_gemms == 0:
        return bias

    device = A[0].device

    # Determine if we have bias
    has_bias = len(bias) > 0 and bias[0].numel() > 0

    # Determine if we have pre_gelu_out
    has_gelu = len(pre_gelu_out) > 0 and pre_gelu_out[0].numel() > 0

    # Compute output shapes
    output_shapes = []
    for i in range(num_gemms):
        M, N, K = _compute_gemm_dimensions(A[i], B[i], transa, transb)
        output_shapes.append((M, N))

    # Handle output tensor allocation
    if single_output:
        # Single output mode - D should contain one tensor
        if D is not None and len(D) > 0:
            out_tensor = D[0]
        else:
            total_M = sum(m_splits) if m_splits else sum(s[0] for s in output_shapes)
            N = output_shapes[0][1]
            out_tensor = torch.empty((total_M, N), dtype=D_type, device=device)

        # Create views for each GEMM
        out_list = []
        start_idx = 0
        for i in range(num_gemms):
            if m_splits and i < len(m_splits):
                size = m_splits[i]
            else:
                size = output_shapes[i][0]
            out_list.append(out_tensor[start_idx : start_idx + size])
            start_idx += size
    else:
        # Discrete output mode
        if D is not None and len(D) > 0:
            out_list = D
        else:
            out_list = [
                torch.empty(shape, dtype=D_type, device=device) for shape in output_shapes
            ]

    # Launch GEMM for each pair
    for i in range(num_gemms):
        # Skip empty tensors
        if A[i].numel() == 0 or B[i].numel() == 0:
            if not accumulate and out_list[i].numel() > 0:
                out_list[i].zero_()
            if has_bias and grad and bias[i].numel() > 0:
                bias[i].zero_()
            if has_gelu and pre_gelu_out[i].numel() > 0:
                pre_gelu_out[i].zero_()
            continue

        _launch_single_gemm(
            A=A[i],
            B=B[i],
            out=out_list[i],
            transa=transa,
            transb=transb,
            bias=bias[i] if has_bias else None,
            pre_gelu_out=pre_gelu_out[i] if has_gelu else None,
            accumulate=accumulate,
        )

    # Return bias (may contain gradients if grad=True)
    return bias


def general_grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    quantization_params: Optional[List] = None,
    out_dtype: torch.dtype = torch.float16,
    layout: str = "TN",
    m_splits: Optional[List[int]] = None,
    gelu: bool = False,
    grad: bool = False,
    accumulate: bool = False,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[torch.dtype] = None,
    single_output: bool = False,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    """
    High-level grouped GEMM API compatible with TransformerEngine's general_grouped_gemm.

    This is a convenience wrapper around te_general_grouped_gemm.
    """
    num_gemms = len(A)
    if num_gemms == 0:
        return [], None, None

    transa = layout[0] == "T"
    transb = layout[1] == "T"

    # Handle D_dtype override
    if D_dtype is not None:
        out_dtype = D_dtype

    device = A[0].device

    # Prepare empty tensors
    empty_tensor = torch.tensor([], device=device)
    empty_tensors = [empty_tensor] * num_gemms

    # Prepare bias
    if use_bias and bias is not None:
        bias_list = bias
        bias_dtype = bias[0].dtype
    else:
        bias_list = empty_tensors
        bias_dtype = torch.bfloat16

    # Prepare grad_bias if needed
    if grad and use_bias:
        # Compute output N dimension
        M, N, K = _compute_gemm_dimensions(A[0], B[0], transa, transb)
        grad_bias = [torch.empty(N, dtype=out_dtype, device=device) for _ in range(num_gemms)]
    else:
        grad_bias = empty_tensors

    # Prepare pre_gelu_out
    if gelu:
        if single_output and m_splits:
            total_M = sum(m_splits)
            N = _compute_gemm_dimensions(A[0], B[0], transa, transb)[1]
            pre_gelu_tensor = torch.empty((total_M, N), dtype=out_dtype, device=device)
            pre_gelu_list = []
            start_idx = 0
            for i, m in enumerate(m_splits):
                pre_gelu_list.append(pre_gelu_tensor[start_idx : start_idx + m])
                start_idx += m
        else:
            pre_gelu_list = [torch.empty_like(o) for o in out]
    else:
        pre_gelu_list = empty_tensors

    # Prepare m_splits
    if m_splits is None:
        m_splits_list = []
    else:
        m_splits_list = m_splits

    # Dummy workspace
    workspace = [torch.zeros(1, dtype=torch.uint8, device=device)]

    # Call te_general_grouped_gemm
    te_general_grouped_gemm(
        A=A,
        transa=transa,
        B=B,
        transb=transb,
        D=out,
        D_type=out_dtype,
        m_splits=m_splits_list,
        bias=grad_bias if grad else bias_list,
        bias_type=bias_dtype,
        single_output=single_output,
        pre_gelu_out=pre_gelu_list,
        grad=grad,
        workspace=workspace,
        workspaceSize=1,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
        math_sm_count=0,
    )

    # Prepare return values
    if single_output and len(out) > 0:
        out_return = out[0]
    else:
        out_return = out

    bias_return = grad_bias if grad else (bias_list if use_bias else None)
    pre_gelu_return = pre_gelu_list if gelu else None

    return out_return, bias_return, pre_gelu_return
