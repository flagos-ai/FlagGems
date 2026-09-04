import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def _apply_2_4_meta(weight, meta):
    """Apply the 2:4 select-position meta to a dense weight.

    ``weight``: (N, K). ``meta``: (N, K // 4) truthy -> keep positions 0,1 of each
    group of 4; falsy -> keep positions 2,3. Returns a dense (N, K) tensor with
    the non-selected positions zeroed, matching the kernel's semantics.
    """
    N, K = weight.shape
    K4 = K // 4
    choice = meta.to(torch.bool)
    wr = weight.view(N, K4, 4)
    keep = torch.cat(
        [
            choice.unsqueeze(2),
            choice.unsqueeze(2),
            (~choice).unsqueeze(2),
            (~choice).unsqueeze(2),
        ],
        dim=2,
    )
    masked = torch.where(keep, wr, torch.zeros_like(wr))
    return masked.reshape(N, K)


def _sparse_semi_structured_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    meta: torch.Tensor,
    bias: torch.Tensor = None,
    activation: str = None,
    out_dtype: torch.dtype = None,
):
    """
    Sparse semi-structured (2:4) linear layer, soft implementation.

    Applies the 2:4 select-position meta to zero out the non-kept positions of
    ``weight`` and then performs a dense matmul, so the result matches the
    Triton kernel's semantics (each weight row carries its own 2:4 pattern).
    """
    logger.debug("GEMS_METAX SPARSE SEMI STRUCTURED LINEAR")

    M, K = input.shape
    N = weight.shape[0]
    K_w = weight.shape[1]

    assert K == K_w, f"Incompatible dimensions: input K={K}, weight K={K_w}"
    assert K % 4 == 0, f"K must be a multiple of 4 for 2:4 sparsity, got K={K}"
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

    # Apply the per-row 2:4 meta, then dense matmul (float32 accumulation).
    masked_weight = _apply_2_4_meta(weight, meta)
    output = torch.matmul(input.float(), masked_weight.t().float())

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
            output = torch.nn.functional.silu(output)
        elif activation == "gelu":
            output = torch.nn.functional.gelu(output)
        else:
            logger.warning(f"Unknown activation: {activation}")

    return output
