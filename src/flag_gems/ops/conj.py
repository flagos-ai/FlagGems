import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def conj_kernel(
    input_ptr,  # 指向形状为 (2, N) 的实数张量，[实部, 虚部]
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载实部（索引 0）
    real = tl.load(input_ptr + offsets, mask=mask)
    # 加载虚部（索引 n_elements）
    imag = tl.load(input_ptr + offsets + n_elements, mask=mask)

    # 存储实部不变
    tl.store(output_ptr + offsets, real, mask=mask)
    # 存储虚部取反
    tl.store(output_ptr + offsets + n_elements, -imag, mask=mask)


def conj(A):
    """Return the conjugate of a complex tensor with physical memory copy.

    This implementation uses a Triton kernel to compute the conjugate
    of a complex tensor, ensuring a physical memory copy.

    Args:
        A (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: A new tensor containing the conjugate of A.
    """
    logger.debug("GEMS CONJ")

    if not isinstance(A, torch.Tensor):
        return torch.tensor(A)

    # 如果输入是实数，直接克隆返回
    if not A.is_complex():
        return A.clone()

    # 确保输入在 GPU 上
    if not A.is_cuda:
        A = A.cuda()

    # 将复数张量转换为实部虚部分开的视图，形状为 (*, 2)
    real_imag = torch.view_as_real(A).contiguous()
    # 将形状从 (*, 2) 转换为 (2, *)
    real_imag_flat = real_imag.permute(-1, *range(real_imag.dim() - 1)).contiguous()
    # 展平为 (2, N)
    real_imag_2d = real_imag_flat.view(2, -1)

    # 创建输出张量
    output_real_imag = torch.empty_like(real_imag_2d)

    n_elements = real_imag_2d.shape[1]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    conj_kernel[grid](
        real_imag_2d,
        output_real_imag,
        n_elements,
        BLOCK_SIZE,
    )

    # 恢复原始形状
    output_flat = output_real_imag.view(real_imag_flat.shape)
    output_real_imag_original = output_flat.permute(*range(1, output_flat.dim()), 0).contiguous()

    return torch.view_as_complex(output_real_imag_original)
