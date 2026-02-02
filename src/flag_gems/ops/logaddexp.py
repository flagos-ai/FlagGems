import triton.language as tl

from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def logaddexp_func_opt(a, b):
    # 1. 统一转为 fp32 计算
    a32 = a.to(tl.float32)
    b32 = b.to(tl.float32)

    # 2. 数学简化：log(exp(a)+exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    m = tl.maximum(a32, b32)
    diff = tl.abs(a32 - b32)
    
    # 3. 性能平衡：
    # 对于 diff 很大的情况，exp(-diff) 趋近于 0，log(1) 趋近于 0
    # 可以通过 tl.where 进一步优化极端情况，但 log1p(exp(-diff)) 通常已经足够快
    # 这里的 tl.math.exp 和 tl.math.log1p 比通用的 log 更快
    res = m + tl.math.log1p(tl.math.exp(-diff))

    return res.to(a.dtype)


def logaddexp(A, B):
    # A y B son tensores CUDA; FlagGems hará el dispatch
    return logaddexp_func(A, B)