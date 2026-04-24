import torch
import triton
import triton.language as tl

# ==============================================================================
# Triton Kernels: 动态启发式与 1D 洪泛网格的完美结合
# ==============================================================================


# 动态探测：超过 20MB (约 5M 个 float32) 才开启 evict_first 压榨 HBM
# 否则保持普通缓存策略 (evict_last)，白嫖 L2 Cache 的极速带宽
@triton.heuristics(
    {
        "EVICT_POLICY": lambda args: "evict_first"
        if args["n_elements"] * 4 > 20_000_000
        else "evict_last"
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=4),
    ],
    key=["n_elements"],
)
@triton.jit
def leaky_relu_fwd_kernel(
    x_ptr,
    y_ptr,
    negative_slope,
    n_elements,
    EVICT_POLICY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Fast-Path: 99.9% 的数据走这里，剔除全局 mask 的 ALU 消耗
    if block_start + BLOCK_SIZE <= n_elements:
        x = tl.load(x_ptr + offsets, eviction_policy=EVICT_POLICY)
        zero = tl.cast(0.0, x.dtype)
        slope = tl.cast(negative_slope, x.dtype)

        # 回归数学上绝对安全的语义，依赖 LLVM IR 的自然展开
        out = tl.where(x > zero, x, x * slope)

        tl.store(y_ptr + offsets, out, eviction_policy=EVICT_POLICY)

    # Tail-Path: 只有最后一个非对齐的 Block 才会执行这部分带 mask 的逻辑
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        zero = tl.cast(0.0, x.dtype)
        slope = tl.cast(negative_slope, x.dtype)

        out = tl.where(x > zero, x, x * slope)

        # 尾部对齐的零碎数据，直接走默认缓存策略
        tl.store(y_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=4),
    ],
    key=["n_elements"],
)
@triton.jit
def leaky_relu_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    grad_in_ptr,
    negative_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    if block_start + BLOCK_SIZE <= n_elements:
        x = tl.load(x_ptr + offsets)
        grad_out = tl.load(grad_out_ptr + offsets)
        zero = tl.cast(0.0, x.dtype)
        slope = tl.cast(negative_slope, x.dtype)
        grad_in = tl.where(x > zero, grad_out, grad_out * slope)
        tl.store(grad_in_ptr + offsets, grad_in)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
        zero = tl.cast(0.0, x.dtype)
        slope = tl.cast(negative_slope, x.dtype)
        grad_in = tl.where(x > zero, grad_out, grad_out * slope)
        tl.store(grad_in_ptr + offsets, grad_in, mask=mask)


# ==============================================================================
# Dispatcher: 异常拦截与内存连续性降级路由
# ==============================================================================
class FlagGemsLeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, negative_slope: float, inplace: bool = False):
        # 拦截 1：非法输入拦截，满足 PR 异常处理规范
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"leaky_relu expected torch.Tensor, but got {type(x)}")
        if not x.is_floating_point():
            raise TypeError(
                f"leaky_relu only supports floating point tensors, but got {x.dtype}"
            )

        # 拦截 2：非连续内存降级
        x_c = x.contiguous() if not x.is_contiguous() else x

        if inplace:
            ctx.mark_dirty(x)
            y = x_c
        else:
            y = torch.empty_like(x_c)

        n_elements = x_c.numel()

        # 1D 洪泛网格，让 GigaThread Engine 全力接管调度
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        # 注意：这里不再需要手动传 EVICT_POLICY，Triton 会通过 heuristics 自动注入
        leaky_relu_fwd_kernel[grid](x_c, y, negative_slope, n_elements)

        if inplace and not x.is_contiguous():
            x.copy_(y)

        ctx.save_for_backward(y if inplace else x_c)
        ctx.negative_slope = float(negative_slope)

        return x if inplace else y.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x_saved,) = ctx.saved_tensors
        grad_out_c = grad_out.contiguous() if not grad_out.is_contiguous() else grad_out
        grad_in = torch.empty_like(grad_out_c)
        n_elements = x_saved.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        leaky_relu_bwd_kernel[grid](
            grad_out_c, x_saved, grad_in, ctx.negative_slope, n_elements
        )

        return grad_in.view_as(grad_out), None, None


def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    Applies the element-wise LeakyReLU function.
    Interface fully aligned with PyTorch.
    """
    return FlagGemsLeakyReLU.apply(x, negative_slope, False)


def leaky_relu_(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """In-place version of leaky_relu."""
    return FlagGemsLeakyReLU.apply(x, negative_slope, True)
