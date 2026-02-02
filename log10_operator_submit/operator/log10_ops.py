import torch
from torch import Tensor
from typing import Optional

class Log10Operator:
    """
    log10 算子实现
    包含三种调用形式：普通版、in-place 版、out 版
    符合 PyTorch 算子接口规范，保证精度和性能
    """
    @staticmethod
    def log10(x: Tensor) -> Tensor:
        """
        普通版 log10 算子：计算以10为底的对数，返回新张量
        Args:
            x: 输入张量（需为正数，否则会触发 PyTorch 原生对数异常）
        Returns:
            Tensor: 与输入同形状/同 dtype 的 log10 计算结果
        """
        # 修复：提前严格校验输入，确保负数/零输入必抛异常
        if (x <= 0).any():
            # 明确提示异常原因，统一异常类型为 ValueError
            raise ValueError(
                f"log10 算子输入必须为严格正数张量！检测到输入包含负数/零：{x[x <= 0]}"
            )
        
        # 核心公式：log10(x) = ln(x) / ln(10)
        ln_10 = torch.log(torch.tensor(10.0, dtype=x.dtype, device=x.device))
        result = torch.log(x) / ln_10
        return result

    @staticmethod
    def log10_(x: Tensor) -> Tensor:
        """
        In-place 版 log10 算子：直接修改输入张量，无额外内存开销
        Args:
            x: 输入张量（需为正数，in-place 修改）
        Returns:
            Tensor: 修改后的原张量
        """
        # 修复：统一输入校验逻辑
        if (x <= 0).any():
            raise ValueError(
                f"log10_ 算子输入必须为严格正数张量！检测到输入包含负数/零：{x[x <= 0]}"
            )
        
        ln_10 = torch.log(torch.tensor(10.0, dtype=x.dtype, device=x.device))
        with torch.no_grad():  # in-place 操作禁用梯度计算，提升性能
            x.log_()  # 原生 in-place 自然对数
            x.div_(ln_10)  # 原生 in-place 除以 ln(10)
        return x

    @staticmethod
    def log10_out(x: Tensor, *, out: Tensor) -> Tensor:
        """
        Out 版 log10 算子：将结果写入指定 out 张量，避免内存分配
        Args:
            x: 输入张量（需为正数）
            out: 输出张量（需与输入同形状/同 dtype/同设备）
        Returns:
            Tensor: 写入结果的 out 张量
        """
        # 修复：统一输入校验逻辑
        if (x <= 0).any():
            raise ValueError(
                f"log10_out 算子输入必须为严格正数张量！检测到输入包含负数/零：{x[x <= 0]}"
            )
        
        # 严格校验 out 张量，符合 PyTorch 原生接口规范
        if out.shape != x.shape:
            raise ValueError(f"out 形状 {out.shape} 与输入 {x.shape} 不匹配")
        if out.dtype != x.dtype:
            raise ValueError(f"out 数据类型 {out.dtype} 与输入 {x.dtype} 不匹配")
        if out.device != x.device:
            raise ValueError(f"out 设备 {out.device} 与输入 {x.device} 不匹配")
        
        ln_10 = torch.log(torch.tensor(10.0, dtype=x.dtype, device=x.device))
        torch.log(x, out=out)  # 原生 out 模式计算自然对数
        out.div_(ln_10)  # in-place 修改 out 张量
        return out

# 对外暴露算子接口
log10 = Log10Operator.log10
log10_ = Log10Operator.log10_
log10_out = Log10Operator.log10_out