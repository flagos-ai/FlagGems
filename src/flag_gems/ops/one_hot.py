import torch

from flag_gems.ops.scatter import scatter_


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    if tensor.dtype != torch.int64:
        raise RuntimeError(
            "one_hot is only applicable to index tensor of type LongTensor."
        )

    if tensor.numel() == 0:
        if num_classes <= 0:
            raise RuntimeError(
                "Can not infer total number of classes from empty tensor."
            )
        shape = (*tensor.shape, num_classes)
        return torch.empty(shape, device=tensor.device, dtype=torch.int64)

    minv = int(tensor.min().item())
    if minv < 0:
        raise RuntimeError("Class values must be non-negative.")
    maxv = int(tensor.max().item())

    if num_classes == -1:
        num_classes = maxv + 1
    else:
        if num_classes < 1:
            raise RuntimeError("num_classes should be positive")
        if maxv >= num_classes:
            raise RuntimeError("Class values must be smaller than num_classes.")

    if tensor.device.type == "cpu":
        out = torch.zeros((*tensor.shape, num_classes), device="cpu", dtype=torch.int64)
        out.scatter_(-1, tensor.unsqueeze(-1), 1)
        return out

    out = torch.zeros(
        (*tensor.shape, num_classes), device=tensor.device, dtype=torch.int64
    )
    index = tensor.unsqueeze(-1)
    src = torch.ones_like(index, dtype=torch.int64)
    scatter_(out, -1, index, src, reduce=None)
    return out
