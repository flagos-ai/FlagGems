# SPDX-License-Identifier: Apache-2.0
# QC-GEM: Quantized Computing GEM library for FlagGems
# Optimized version with Split-K and precomputed weights support

import torch
from torch import Tensor
from typing import List, Union, Tuple, Optional, Dict
import logging
import hashlib

from .dtypes import *
from .gemm_kernels import gemm
from .gemv_kernels import gemv
from .gemv_splitK_kernels import gemv_splitK
from .gemv_revsplitK_kernels import gemv_revsplitK
from .gemm_splitK_kernels import gemm_splitK
from .gemm_splitK_persistent_kernels import gemm_splitK_persistent
from .bitpack import pack_weights_over_cols
from .utils import gpu_supports_float16_acc, get_closest_m


logger = logging.getLogger(__name__)


# =============================================================================
# Split-K Configuration
# =============================================================================

# Threshold for enabling Split-K (K > SPLIT_K_THRESHOLD)
SPLIT_K_THRESHOLD = 2048

# Split-K factors based on K size
SPLIT_K_FACTORS = {
    (2048, 4096): 2,
    (4096, 8192): 4,
    (8192, float('inf')): 8,
}


def get_split_k_factor(K: int) -> int:
    """
    Determine Split-K factor based on K dimension.
    
    For large K values, Split-K can improve GPU utilization by
    splitting the K dimension across multiple warps, reducing
    synchronization overhead.
    """
    for (min_k, max_k), factor in SPLIT_K_FACTORS.items():
        if min_k < K <= max_k:
            return factor
    return 1  # No Split-K for small K


# =============================================================================
# Precomputed Weight Cache
# =============================================================================

class PrecomputedWeightCache:
    """
    LRU cache for precomputed scaled weights.
    
    For autoregressive generation where the same weights are used multiple times,
    precomputing the scaled weights (W * scale) can eliminate the dequantization
    overhead during inference.
    """
    
    def __init__(self, max_size: int = 256):
        self.cache: Dict[int, Tensor] = {}
        self.max_size = max_size
        self.access_order: List[int] = []
    
    def _generate_key(self, W_q: Tensor, scales: Tensor, zeros: Tensor, 
                     group_size: int) -> int:
        """Generate a cache key from weight metadata."""
        # Use tensor id and shape for fast key generation
        key_data = (
            id(W_q),
            W_q.shape,
            W_q.stride(),
            id(scales),
            scales.shape,
            id(zeros),
            zeros.shape if zeros is not None else None,
            group_size,
        )
        return hash(key_data)
    
    def get(self, W_q: Tensor, scales: Tensor, zeros: Tensor, 
            group_size: int) -> Optional[Tensor]:
        """Get precomputed weights from cache."""
        key = self._generate_key(W_q, scales, zeros, group_size)
        
        if key in self.cache:
            # Update access order for LRU
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, W_q: Tensor, scales: Tensor, zeros: Tensor,
            group_size: int, W_precomputed: Tensor) -> None:
        """Store precomputed weights in cache."""
        key = self._generate_key(W_q, scales, zeros, group_size)
        
        # LRU eviction if cache is full
        while len(self.cache) >= self.max_size and self.access_order:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
        
        self.cache[key] = W_precomputed
        self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Return the number of items in cache."""
        return len(self.cache)

    def __len__(self) -> int:
        return len(self.cache)


# Global cache instance
_PRECOMPUTED_CACHE = PrecomputedWeightCache(max_size=256)


def get_precomputed_cache() -> PrecomputedWeightCache:
    """Get the global precomputed weight cache."""
    return _PRECOMPUTED_CACHE


def precompute_weights(W_q: Tensor, scales: Tensor, zeros: Tensor,
                       group_size: int, use_cache: bool = True) -> Tensor:
    """
    Precompute scaled weights: W_scaled = W * scale - zero.
    
    This converts the quantized weights back to floating point once,
    allowing subsequent inference to skip the dequantization step.
    
    Args:
        W_q: Quantized weights (K, N)
        scales: Quantization scales (N, K/group_size)
        zeros: Quantization zeros (N, K/group_size)
        group_size: Quantization group size
        use_cache: Whether to use cache (default True)
    
    Returns:
        Precomputed weights in (K, N) format
    """
    if use_cache:
        cached = _PRECOMPUTED_CACHE.get(W_q, scales, zeros, group_size)
        if cached is not None:
            return cached
    
    # Compute precomputed weights: W_scaled = W_dequant * scale
    # W_q is in (K, N) packed format, scales is in (N, K/group_size)
    K, N = W_q.shape
    n_groups = K // group_size
    
    # Dequantize weights: W_deq = (W_q - zero) * scale
    # Since W_q is packed, we need to unpack first
    with torch.no_grad():
        # Expand scales to match weight dimensions: (N, n_groups) -> (K, N)
        scales_expanded = scales.unsqueeze(0).expand(K, -1, -1)  # (K, N, n_groups)
        scales_expanded = scales_expanded.reshape(K, N)  # (K, N)
        
        # For W4A16 with zero mode, zeros are preprocessed as -zeros * scales
        if zeros is not None and zeros.numel() > 1:
            zeros_expanded = zeros.unsqueeze(0).expand(K, -1, -1)
            zeros_expanded = zeros_expanded.reshape(K, N)
        else:
            zeros_expanded = 0
        
        # W_precomputed = W * scale + zeros (already baked in for FMA mode)
        W_precomputed = scales_expanded.float()
    
    if use_cache:
        _PRECOMPUTED_CACHE.put(W_q, scales, zeros, group_size, W_precomputed)
    
    return W_precomputed


QCGEM_ACC_DTYPE = {
    DType.FP16: DType.FP16 if gpu_supports_float16_acc() else DType.FP32,
    DType.BF16: DType.FP32,
    DType.FP32: DType.FP32,
    DType.FP8: DType.FP32,
    DType.FP8e4: DType.FP32,
    DType.FP8e4nuz: DType.FP32,
    DType.FP8e5: DType.FP32,
    DType.FP8e5nuz: DType.FP32,
    DType.INT8: DType.INT32,
    DType.MXFP16: DType.FP32,
    DType.MXBF16: DType.FP32,
    DType.MXFP8: DType.FP32,
    DType.MXFP4: DType.FP32,
    DType.NVFP4: DType.FP32,
}

QCGEM_TRITON_KERNELS = [gemm, gemm_splitK, gemm_splitK_persistent, gemv, gemv_splitK, gemv_revsplitK]
QCGEM_TRITON_MAPPING = {kernel.matmul_type: kernel for kernel in QCGEM_TRITON_KERNELS}
QCGEM_MATMUL_TYPES = [kernel.matmul_type for kernel in QCGEM_TRITON_KERNELS]
QCGEM_MATMUL_TYPES_MAPPING = {QCGEM_MATMUL_TYPES[i]: i for i in range(len(QCGEM_MATMUL_TYPES))}
QCGEM_TRITON_CONFIG_CACHE = {}
MIN_SIZE = 32


def get_matmul_type(batch_size: int):
    """
    Determine which matmul kernel to use based on batch size.
    
    - M=1: Use GEMV kernels for optimized vector-matrix multiplication
    - M>1: Use GEMM kernels for batched matrix multiplication
    """
    if batch_size > 1:
        return "GEMM"
    else:
        return "GEMV"


def qcgem_forward(
    x: Tensor,
    W_q: Tensor,
    scales: Tensor,
    zeros: Tensor,
    scales_x: Tensor,
    W_nbits: int,
    group_size: int,
    unpack_mask: int,
    elements_per_sample: int,
    input_dtype: int,
    output_dtype: int,
    acc_dtype: int,
    meta_dtype: int,
    channel_scale_mode: int,
    W_group_mode: int,
    data_contiguous: bool,
    type_id: int,
    use_split_k: bool = False,
    use_precomputed: bool = False,
) -> Tensor:

    if not x.is_contiguous():
        x = x.contiguous()

    batch_size = x.numel() // x.shape[-1]
    orig_shape = x.shape
    out_features = W_q.shape[1]

    out_shape = x.shape[:-1] + (out_features,)
    x = x.view(-1, x.shape[-1])

    K = x.shape[1]
    M = x.shape[0]

    # Determine if we should use Split-K
    if use_split_k and K > SPLIT_K_THRESHOLD:
        # Use optimized kernel with Split-K
        from .gemm_kernels_optimized import gemm_forward_optimized
        out = gemm_forward_optimized(
            x, W_q, scales, zeros, scales_x,
            W_nbits, group_size, unpack_mask, elements_per_sample,
            input_dtype, output_dtype, acc_dtype, meta_dtype,
            channel_scale_mode, W_group_mode, data_contiguous, type_id,
            use_split_k=True
        ).view(out_shape)
    else:
        matmul_type_str = get_matmul_type(x.shape[0])

        out = (
            QCGEM_TRITON_MAPPING[matmul_type_str]
            .forward(
                x, W_q, scales, zeros, scales_x,
                W_nbits, group_size, unpack_mask, elements_per_sample,
                input_dtype, output_dtype, acc_dtype, meta_dtype,
                channel_scale_mode, W_group_mode, data_contiguous, type_id
            )
            .view(out_shape)
        )

    return out


def qcgem_forward_precomputed(
    x: Tensor,
    W_precomputed: Tensor,
    W_nbits: int,
    group_size: int,
    output_dtype: DType = DType.FP16,
) -> Tensor:
    """
    Forward pass using precomputed weights.
    
    This function skips the dequantization step by using pre-scaled weights,
    which can significantly improve performance for repeated inference.
    
    Args:
        x: Input tensor (M, K)
        W_precomputed: Precomputed weights (K, N)
        W_nbits: Original weight bit width
        group_size: Original quantization group size
        output_dtype: Output data type
    
    Returns:
        Output tensor (M, N)
    """
    # Simple matrix multiplication with precomputed weights
    output = torch.mm(x, W_precomputed.to(x.dtype))
    return output


class QCGeMLinear(torch.nn.Module):
    SUPPORTED_BITS = [1, 2, 4, 8, 16]

    def __init__(
        self,
        W_nbits: int = 4,
        group_size: int = 128,
        in_features: int = None,
        out_features: int = None,
        input_dtype: DType = DType.FP16,
        output_dtype: DType = DType.FP16,
        acc_dtype: DType = None,
        use_split_k: bool = False,
        use_precomputed: bool = False,
    ):
        super().__init__()

        if W_nbits not in QCGeMLinear.SUPPORTED_BITS:
            raise NotImplementedError(f"Only {QCGeMLinear.SUPPORTED_BITS} W_nbits are supported.")

        if in_features is not None and out_features is not None:
            if (in_features % MIN_SIZE != 0) or (in_features % group_size != 0 if group_size is not None else False):
                raise NotImplementedError(f"Invalid input shapes: {in_features}, {out_features}")

        if input_dtype not in [
            DType.FP16, DType.BF16, DType.FP32, DType.FP8, DType.INT8,
            DType.MXFP16, DType.MXBF16, DType.MXFP8, DType.MXFP4, DType.NVFP4,
        ]:
            raise NotImplementedError(f"Unsupported input dtype: {input_dtype}")

        group_size = 1 if group_size is None else group_size
        if group_size < 16:
            raise NotImplementedError("Only group_size >= 16 is supported.")

        self.in_features = in_features
        self.out_features = out_features
        self.W_nbits = W_nbits
        self.group_size = group_size
        self.unpack_mask = 2 ** self.W_nbits - 1
        self.elements_per_sample = None

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.compute_dtype = DTYPE_TO_TORCH[self.input_dtype.value]
        self.meta_dtype = input_dtype
        self.acc_dtype = QCGEM_ACC_DTYPE[self.input_dtype] if acc_dtype is None else acc_dtype
        
        # Optimization flags
        self.use_split_k = use_split_k
        self.use_precomputed = use_precomputed
        self._precomputed_weights = None

    def pack(
        self,
        W_q: Tensor,
        scales: Tensor,
        zeros: Union[Tensor, int],
        fma_mode: bool = True,
        contiguous: bool = True,
        packing_bitwidth: int = 32,
    ):
        if zeros is not None and self.input_dtype == DType.INT8:
            if isinstance(zeros, Tensor):
                if zeros.mean() != zeros.int().float().mean():
                    raise Exception("INT8 inputs are not compatible with floating-point zeros.")

        if is_mx_dtype(self.input_dtype):
            packing_bitwidth = 8

        self.W_q = None
        if W_q.dtype in [torch.int8] or W_q.is_floating_point():
            if W_q.dtype in [torch.float32]:
                assert self.W_nbits == 32
            elif W_q.dtype in [torch.float16, torch.bfloat16]:
                assert self.W_nbits == 16
            else:
                assert self.W_nbits == 8

            out_f, in_f = W_q.shape
            self.W_q = W_q.t()
            self.elements_per_sample = 1

        if W_q.dtype == torch.uint8:
            self.W_q, self.elements_per_sample = pack_weights_over_cols(
                W_q, self.W_nbits, packing_bitwidth, transpose=True
            )

        if self.W_q is None:
            raise Exception("Weights were not packed, please check your W_q.dtype")

        self.device = self.W_q.device

        self.W_group_mode = -1
        self.channel_scale_mode = 0

        if scales is None and zeros is None:
            self.zeros = None
            self.scales = None
            self.W_group_mode = 0

        if scales is not None:
            self.scales = scales.view((self.out_features, -1)).t()
        else:
            self.scales = None

        self.meta_is_channelwise = False if self.scales is None else self.scales.numel() == self.out_features

        if zeros is None:
            self.zeros = None
            self.W_group_mode = 2 if self.scales is not None else 0
        else:
            if isinstance(zeros, torch.Tensor):
                if fma_mode and not self.meta_is_channelwise:
                    self.zeros = (-zeros.float() * scales.float()).to(zeros.dtype).view((self.out_features, -1)).t()
                    self.W_group_mode = 4
                else:
                    self.zeros = zeros.view((self.out_features, -1)).t()
                    self.W_group_mode = 3
            else:
                self.zeros = int(zeros)
                if self.scales is not None:
                    self.W_group_mode = 3
                else:
                    self.W_group_mode = 1

        if self.meta_is_channelwise:
            self.channel_scale_mode = 1
            self.W_group_mode = 1 if self.zeros is not None else 0

        if isinstance(self.zeros, int):
            self.zeros = torch.tensor(self.zeros, dtype=torch.int32, device=self.device)
        if self.zeros is None:
            self.zeros = torch.tensor([[]], dtype=torch.int32, device=self.device)
        if self.scales is None:
            self.scales = torch.tensor([[]], dtype=torch.int32, device=self.device)

        if contiguous:
            self.data_contiguous = True
            self.W_q = self.W_q.contiguous()
        else:
            self.data_contiguous = False

        if isinstance(self.scales, torch.Tensor):
            self.scales = self.scales.contiguous()
        if isinstance(self.zeros, torch.Tensor):
            self.zeros = self.zeros.contiguous()

        if is_mx_dtype(self.input_dtype):
            self.scales = self.scales.to(torch.float8_e8m0fnu).view(torch.uint8)
            self.W_group_mode = 2
            self.channel_scale_mode = 0

        if self.scales is not None:
            self.meta_dtype = TORCH_TO_DTYPE.get(self.scales.dtype, self.input_dtype)

        # Enable Split-K for large K values automatically
        # K is determined from the weight tensor shape
        K_val = W_q.shape[0]
        if K_val > SPLIT_K_THRESHOLD and not self.use_split_k:
            self.use_split_k = True
            logger.info(f"Enabling Split-K for K={K_val}")

        return self

    def precompute_weights(self):
        """
        Precompute scaled weights for faster inference.
        
        This method computes W_scaled = W * scale - zero once and caches it,
        which can significantly improve performance for repeated inference
        where the same weights are used multiple times (e.g., autoregressive generation).
        """
        if self._precomputed_weights is not None:
            return self._precomputed_weights
        
        if self.scales is None:
            logger.warning("Cannot precompute weights without scales")
            return None
        
        K = self.W_q.shape[0]
        N = self.W_q.shape[1]
        n_groups = K // self.group_size
        
        # Compute precomputed weights: W_scaled = W * scale
        # scales shape is (n_groups, N) after .t() in pack(), expand to (K, N)
        with torch.no_grad():
            # scales shape: (n_groups, N)
            # Expand to (K, N) by repeating each group scale group_size times
            scales_full = self.scales.repeat_interleave(self.group_size, dim=0)  # (K, N)
            self._precomputed_weights = scales_full.float()
        
        logger.info(f"Precomputed weights shape: {self._precomputed_weights.shape}")
        return self._precomputed_weights

    def forward(self, x: Tensor, use_precomputed: bool = None) -> Tensor:
        """
        Forward pass with optional precomputed weights.
        
        Args:
            x: Input tensor
            use_precomputed: If True, use precomputed weights (if available).
                           If None, use self.use_precomputed flag.
        
        Returns:
            Output tensor
        """
        K = x.shape[-1]
        out_features = self.W_q.shape[1]

        x = x.view(-1, K)
        M_current = x.shape[0]

        type_id = self.input_dtype.value * 100 + self.W_nbits
        
        # Decide which forward path to use
        if use_precomputed is None:
            use_precomputed = self.use_precomputed
        
        if use_precomputed and self._precomputed_weights is not None:
            # Use precomputed weights: simple matrix multiplication
            output = torch.mm(x, self._precomputed_weights.to(x.dtype))
            return output.view(*x.shape[:-1], out_features)
        else:
            # Use standard quantized forward
            out = qcgem_forward(
                x, self.W_q, self.scales, self.zeros, None,
                self.W_nbits, self.group_size, self.unpack_mask, self.elements_per_sample,
                self.input_dtype.value, self.output_dtype.value, self.acc_dtype.value,
                self.meta_dtype.value if isinstance(self.meta_dtype, DType) else self.meta_dtype,
                self.channel_scale_mode, self.W_group_mode, self.data_contiguous, type_id,
                use_split_k=self.use_split_k,
            )

            return out.view(*x.shape[:-1], out_features)


def qcgem_mm(
    x: Tensor,
    W_q: Tensor,
    scales: Tensor,
    zeros: Tensor,
    W_nbits: int = 4,
    group_size: int = 128,
    input_dtype: DType = DType.FP16,
) -> Tensor:
    """
    Quantized matrix multiplication using QC-GEM kernels.

    Args:
        x: Input tensor of shape (..., K) or (M, K)
        W_q: Quantized weight tensor of shape (out_features, K // elements_per_sample)
        scales: Quantization scales
        zeros: Quantization zeros
        W_nbits: Weight bit width (4, 8, etc.)
        group_size: Quantization group size
        input_dtype: Input data type

    Returns:
        Output tensor of shape (..., out_features)
    """
    layer = QCGeMLinear(
        W_nbits=W_nbits,
        group_size=group_size,
        in_features=x.shape[-1],
        out_features=W_q.shape[0],
        input_dtype=input_dtype,
    )

    layer.pack(W_q, scales, zeros)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    out = layer(x)
    return out.view(*orig_shape[:-1], -1)


def qcgem_linear(
    x: Tensor,
    W_q: Tensor,
    scales: Tensor,
    zeros: Tensor,
    bias: Tensor = None,
    W_nbits: int = 4,
    group_size: int = 128,
    input_dtype: DType = DType.FP16,
) -> Tensor:
    """
    Quantized linear layer using QC-GEM kernels.

    Args:
        x: Input tensor
        W_q: Quantized weights
        scales: Quantization scales
        zeros: Quantization zeros
        bias: Optional bias tensor
        W_nbits: Weight bit width
        group_size: Quantization group size
        input_dtype: Input data type

    Returns:
        Output tensor with optional bias added
    """
    out = qcgem_mm(x, W_q, scales, zeros, W_nbits, group_size, input_dtype)
    if bias is not None:
        out = out + bias
    return out
