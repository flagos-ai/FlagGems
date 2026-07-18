# src/flag_gems/ops/meshgrid.py
import torch


def meshgrid(tensors, indexing="ij"):
    """
    FlagGems 高性能 meshgrid 实现

    针对不同维度采用最优策略：
    - 2D: 使用 view + expand（最快）
    - 3D: 使用 view + expand（次快）
    - 4D+: 使用 view + broadcast_tensors（通用，PyTorch C++ 加速）

    Args:
        tensors: 一维张量列表
        indexing: 'ij' 或 'xy'

    Returns:
        tuple of tensors
    """
    if not tensors:
        return ()

    if indexing not in ["ij", "xy"]:
        raise ValueError(f"indexing must be 'ij' or 'xy', got {indexing}")

    # ============ 快速检测 NPU ============
    device = tensors[0].device
    if device.type == "npu":
        return torch.meshgrid(tensors, indexing=indexing)

    ndim = len(tensors)

    # 确保所有输入是 1D（减少后续检查开销）
    for i, t in enumerate(tensors):
        if t.dim() != 1:
            tensors[i] = t.flatten()

    # ============ 2D 快速路径 ============
    if ndim == 2:
        x, y = tensors[0], tensors[1]
        if indexing == "ij":
            return (
                x.view(-1, 1).expand(x.size(0), y.size(0)),
                y.view(1, -1).expand(x.size(0), y.size(0)),
            )
        else:  # xy
            return (
                x.view(1, -1).expand(y.size(0), x.size(0)),
                y.view(-1, 1).expand(y.size(0), x.size(0)),
            )

    # ============ 3D 快速路径 ============
    if ndim == 3:
        x, y, z = tensors[0], tensors[1], tensors[2]
        if indexing == "ij":
            return (
                x.view(-1, 1, 1).expand(x.size(0), y.size(0), z.size(0)),
                y.view(1, -1, 1).expand(x.size(0), y.size(0), z.size(0)),
                z.view(1, 1, -1).expand(x.size(0), y.size(0), z.size(0)),
            )
        else:  # xy
            return (
                x.view(1, -1, 1).expand(y.size(0), x.size(0), z.size(0)),
                y.view(-1, 1, 1).expand(y.size(0), x.size(0), z.size(0)),
                z.view(1, 1, -1).expand(y.size(0), x.size(0), z.size(0)),
            )

    # ============ 4D+ 通用路径 ============
    # 使用 view + broadcast_tensors（高效 C++ 实现）
    if indexing == "ij":
        # ij 模式：第 i 个张量在第 i 维展开
        reshaped = []
        for i, t in enumerate(tensors):
            # 构建 shape: [1, 1, ..., n, 1, ..., 1]
            shape = [1] * ndim
            shape[i] = -1  # -1 自动推断
            reshaped.append(t.view(*shape))
        return torch.broadcast_tensors(*reshaped)
    else:  # xy
        # xy 模式：前两个维度交换
        if ndim >= 2:
            reshaped = []
            for i, t in enumerate(tensors):
                shape = [1] * ndim
                if i == 0:
                    shape[1] = -1  # x 在第二维
                elif i == 1:
                    shape[0] = -1  # y 在第一维
                else:
                    shape[i] = -1  # 其他维度
                reshaped.append(t.view(*shape))
            return torch.broadcast_tensors(*reshaped)
        else:
            # ndim == 1，直接返回
            return tensors


def register_ops(registry):
    """注册算子到 PDU"""
    registry.register_op("meshgrid", meshgrid, "aten::meshgrid")
