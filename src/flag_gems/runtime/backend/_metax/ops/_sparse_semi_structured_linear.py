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
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"Unsupported dtype: {input.dtype}"

    # Determine output dtype
    if out_dtype is not None:
        output_dtype = out_dtype
    else:
        output_dtype = input.dtype

    # Compute: output = input @ weight.T (like nn.Linear)
    # Use float32 accumulation for consistency with the triton kernel.
    output = torch.matmul(input.float(), weight.t().float())

    # Add bias if provided (converted to float32 for consistency)
    if bias is not None:
        output = output + bias.float()

    logger.debug(
        "GEMS_METAX SPARSE SEMI STRUCTURED LINEAR, [shape info]: [-, %s, %s, %s](batch, M, N, K)",
        M,
        N,
        K,
    )

    # Convert to output dtype
    if output_dtype != torch.float32:
        output = output.to(output_dtype)

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
