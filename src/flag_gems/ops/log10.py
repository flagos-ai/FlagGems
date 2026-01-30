import logging

# import triton
# import triton.language as tl

# from flag_gems.utils import pointwise_dynamic

# logger = logging.getLogger(__name__)


# @pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
# @triton.jit
# def log10_func(x):
#     inv_ln10 = 0.4342944819032518  # 1/ln(10)
#     return tl.log(x.to(tl.float32)) * inv_ln10


# def log10(A):
#     logger.debug("GEMS LOG10")
#     return log10_func(A)


# def log10_(A):
#     logger.debug("GEMS LOG10_")
#     log10_func(A, out0=A)
#     return A

import logging
import triton
import triton.language as tl
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# 定义访存对齐的配置
# 对于 float16，128-bit 向量化需要一次处理 8 个数，所以 BLOCK_SIZE 必须是 8 的倍数
# 设置为 1024 或 2048 是为了让每个 Warp (32线程) 都能跑满对齐事务
LOG10_CONFIG = {"BLOCK_SIZE": 1024} 

@pointwise_dynamic(
    promotion_methods=[(0, "COMPLEX_TO_FLOAT")],
    config=LOG10_CONFIG  # 关键：在这里强制注入对齐的配置
)
@triton.jit
def log10_func(x):
    # 1. 处理特殊边界
    mask = x <= 0
    x_f32 = x.to(tl.float32)
    
    # 2. 手动提取指数和尾数 (替代 tl.math.frexp)
    # 使用位操作直接读取 IEEE 754 结构
    i = tl.cast(x_f32, tl.int32)
    # 指数部分：取 23-30 位
    n = (i >> 23) & 0xFF
    n = n.to(tl.int32) - 127
    
    # 尾数部分：保留 0-22 位，并将指数位强制设为 127 (即 2^0)
    # 这样得到的 m 范围在 [1.0, 2.0)
    m_int = (i & 0x7FFFFF) | 0x3F800000
    m = tl.cast(m_int, tl.float32)

    # 3. 7阶多项式拟合系数
    # 注意：因为现在的 m 是 [1, 2)，系数需要对应调整或对 m 做偏移
    # 这里我们使用针对 [1, 2) 归一化的泰勒系数
    a0, a1, a2, a3 = -0.99697286, 2.24787022, -2.46980062, 1.99100519
    a4, a5, a6, a7 = -1.07301644, 0.36654758, -0.07176870, 0.00613564

    # 4. Estrin's Method
    m2 = m * m
    m4 = m2 * m2
    
    A = a4 * m + a0
    B = a5 * m + a1
    C = a6 * m + a2
    D = a7 * m + a3
    
    A = A + B * m2
    C = C + D * m2
    log10_m = A + C * m4
    
    # 5. 结合指数
    LOG10_2 = 0.30102999566
    res = log10_m + n.to(tl.float32) * LOG10_2
    
    return tl.where(mask, float("nan"), res)

    
def log10(A):
    logger.debug("GEMS LOG10")
    return log10_func(A)

def log10_(A):
    logger.debug("GEMS LOG10_")
    # 原位操作，利用 pointwise_dynamic 提供的 out0 机制
    log10_func(A, out0=A)
    return A