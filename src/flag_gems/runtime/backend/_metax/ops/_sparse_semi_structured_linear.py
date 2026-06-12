import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def _sparse_semi_structured_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    meta: torch.Tensor,
    bias: torch.Tensor = None,
    activation: str = None,
    out_dtype: torch.dtype = None,
):
    """
    Sparse semi-structured linear layer.

    This implementation treats the sparse weight as dense for the matmul.
    The meta tensor is provided for API compatibility but the actual sparse
    parsing is not implemented (CUTLASS required for proper sparse support).
    """
    logger.debug("GEMS_METAX SPARSE SEMI STRUCTURED LINEAR")

    M, K = input.shape
    N = weight.shape[0]
    K_w = weight.shape[1]

    assert K == K_w, f"Incompatible dimensions: input K={K}, weight K={K_w}"

    # Compute: output = input @ weight.T (like nn.Linear)
    output = torch.matmul(input, weight.t())

    # Apply output dtype if specified
    if out_dtype is not None:
        output = output.to(out_dtype)

    # Add bias if provided
    if bias is not None:
        output = output + bias

    logger.debug(
        "GEMS_METAX SPARSE SEMI STRUCTURED LINEAR, [shape info]: [-, %s, %s, %s](batch, M, N, K)",
        M,
        N,
        K,
    )

    # Apply activation if specified
    if activation is not None:
        if activation == "relu":
            output = torch.relu(output)
        elif activation == "silu" or activation == "swish":
            output = torch.silu(output)
        elif activation == "gelu":
            output = torch.gelu(output)
        else:
            logger.warning(f"Unknown activation: {activation}")

    return output
