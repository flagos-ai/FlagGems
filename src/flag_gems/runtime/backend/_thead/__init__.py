"""
T-Head Zhenwu (真武) PPU Backend Configuration

Product: Zhenwu PPU (真武处理器)
- Model: Zhenwu 810E (supports up to 16 cards with ICN interconnect)
- Architecture: Proprietary T-Head AI accelerator architecture
- SDK: PPU SDK v2.0.0+

Key Features:
- Full CUDA API compatibility (cuda runtime & driver APIs)
- Triton support: 2.3.x, 3.0.x - 3.4.x with AIU extensions
- Accelerated libraries: acdnn, acblas, acfft, acsolver, acrand, acsparse
- Multi-card support: ICN interconnect, MIG (up to 8 instances), SRIOV
- Device management: ppu-smi tool (similar to nvidia-smi)

Hardware Capabilities:
- Tensor Core support with extended PTX instructions
- Dynamic frequency scaling (200MHz ~ max frequency)
- Support for FP16/BF16/FP32/INT8 precision
- High-bandwidth memory with optimized access patterns

PyTorch Integration:
- Uses torch.cuda interface (CUDA-compatible API)
- Compatible with existing CUDA-based PyTorch code
- No special torch.ppu module required

Reference:
- Official Documentation: https://help.aliyun.com/zh/document_detail/3011255.html
"""

from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name="thead",
    # PPU uses CUDA-compatible API, accessed via torch.cuda
    device_name="cuda",
    # PPU device management tool (similar to nvidia-smi)
    device_query_cmd="ppu-smi",
    # Use standard CUDA dispatch key
    dispatch_key=None,
    # PPU has custom Triton backend with AIU extensions
    # The compiler supports Triton 2.3.x - 3.4.x
    triton_extra_name=None,  # Uses standard CUDA path with PPU-specific compiler
)

# Operators that should use PyTorch native implementation
# Based on PPU SDK capabilities and performance characteristics
CUSTOMIZED_UNUSED_OPS = (
    # PPU has strong acceleration library support (acdnn, acblas, etc.)
    # Most operators should benefit from FlagGems optimization
    # This list can be tuned based on benchmarking results
)

__all__ = ["*"]
