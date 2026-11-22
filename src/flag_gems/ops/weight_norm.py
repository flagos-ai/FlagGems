import torch


def weight_norm_interface(w: torch.Tensor, g: torch.Tensor, dim: int) -> torch.Tensor:
    """
    WeightNorm forward: w_norm = g * w / norm(w, dim=dim, keepdim=True)
    占位实现，后续替换为triton fused kernel
    """
    norm = torch.norm(w, dim=dim, keepdim=True)
    return g * w / norm


def weight_norm_interface_backward(
    grad_out: torch.Tensor, w: torch.Tensor, g: torch.Tensor, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    WeightNorm backward，占位实现
    return (grad_w, grad_g)
    """
    norm = torch.norm(w, dim=dim, keepdim=True)
    w_normalized = w / norm
    grad_g = (grad_out * w_normalized).sum(dim=dim, keepdim=True)
    grad_w = g / norm * (grad_out - grad_g * w_normalized)
    return grad_w, grad_g
