import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _parse_einsum_equation(
    equation: str,
) -> Tuple[List[str], Optional[str], bool]:
    """
    Parse einsum equation into input subscripts and output subscripts.

    Args:
        equation: einsum equation string (e.g., "ij,jk->ik")

    Returns:
        Tuple of (input_subscripts_list, output_subscripts, has_ellipsis)
    """
    equation = equation.replace(" ", "")

    has_ellipsis = "..." in equation

    if "->" in equation:
        inputs_str, output = equation.split("->")
    else:
        inputs_str = equation
        output = None

    input_subscripts = inputs_str.split(",")

    return input_subscripts, output, has_ellipsis


def _is_matmul_pattern(input_subscripts: List[str], output: Optional[str]) -> bool:
    """Check if the equation represents a simple matrix multiplication."""
    if len(input_subscripts) != 2:
        return False
    if output is None:
        return False

    s1, s2 = input_subscripts
    # Pattern: "ij,jk->ik" or similar
    if len(s1) == 2 and len(s2) == 2 and len(output) == 2:
        # Check if there's exactly one shared index
        set1 = set(s1)
        set2 = set(s2)
        shared = set1 & set2
        if len(shared) == 1:
            shared_idx = shared.pop()
            # Shared index should be the last in s1 and first in s2
            if s1[1] == shared_idx and s2[0] == shared_idx:
                return True
    return False


def _is_bmm_pattern(input_subscripts: List[str], output: Optional[str]) -> bool:
    """Check if the equation represents batch matrix multiplication."""
    if len(input_subscripts) != 2:
        return False
    if output is None:
        return False

    s1, s2 = input_subscripts
    # Pattern: "bij,bjk->bik" or similar with batch dimensions
    if len(s1) >= 3 and len(s2) >= 3 and len(output) >= 3:
        # Check batch dimensions match
        batch_dims_1 = s1[:-2]
        batch_dims_2 = s2[:-2]
        batch_dims_out = output[:-2]

        if batch_dims_1 == batch_dims_2 == batch_dims_out:
            # Check matrix dimensions
            m1, k1 = s1[-2:]
            k2, n2 = s2[-2:]
            m_out, n_out = output[-2:]

            if k1 == k2 and m1 == m_out and n2 == n_out:
                return True
    return False


def _is_dot_pattern(input_subscripts: List[str], output: Optional[str]) -> bool:
    """Check if the equation represents a dot product."""
    if len(input_subscripts) != 2:
        return False

    s1, s2 = input_subscripts
    # Pattern: "i,i->" or "i,i" (sum of element-wise product)
    if len(s1) == 1 and len(s2) == 1 and s1 == s2:
        if output is None or output == "":
            return True
    return False


def _is_outer_pattern(input_subscripts: List[str], output: Optional[str]) -> bool:
    """Check if the equation represents an outer product."""
    if len(input_subscripts) != 2:
        return False
    if output is None:
        return False

    s1, s2 = input_subscripts
    # Pattern: "i,j->ij"
    if len(s1) == 1 and len(s2) == 1 and len(output) == 2:
        set1 = set(s1)
        set2 = set(s2)
        if not (set1 & set2) and output == s1 + s2:
            return True
    return False


def _is_trace_pattern(input_subscripts: List[str], output: Optional[str]) -> bool:
    """Check if the equation represents a trace operation."""
    if len(input_subscripts) != 1:
        return False

    s1 = input_subscripts[0]
    # Pattern: "ii->" or "ii" for trace
    if len(s1) == 2 and s1[0] == s1[1]:
        if output is None or output == "":
            return True
    return False


def _is_diagonal_pattern(input_subscripts: List[str], output: Optional[str]) -> bool:
    """Check if the equation represents a diagonal extraction."""
    if len(input_subscripts) != 1:
        return False
    if output is None:
        return False

    s1 = input_subscripts[0]
    # Pattern: "ii->i" for diagonal
    if len(s1) == 2 and s1[0] == s1[1] and len(output) == 1 and output[0] == s1[0]:
        return True
    return False


def _is_transpose_pattern(input_subscripts: List[str], output: Optional[str]) -> bool:
    """Check if the equation represents a transpose/permutation."""
    if len(input_subscripts) != 1:
        return False
    if output is None:
        return False

    s1 = input_subscripts[0]
    # Check if output is a permutation of input
    if set(s1) == set(output) and len(s1) == len(output):
        # It's a permutation if output differs from input
        return s1 != output
    return False


def _is_ellipsis_bmm_pattern(equation: str) -> bool:
    """Check if the equation is an ellipsis batch matrix multiplication pattern."""
    # Pattern: "...ij,...jk->...ik" or similar
    equation = equation.replace(" ", "")
    if "->" not in equation:
        return False
    inputs_str, output = equation.split("->")
    operands = inputs_str.split(",")
    if len(operands) != 2:
        return False

    s1, s2 = operands
    # Check for ellipsis pattern
    if not (s1.startswith("...") and s2.startswith("...") and output.startswith("...")):
        return False

    # Extract subscripts after ellipsis
    s1_sub = s1[3:]  # after "..."
    s2_sub = s2[3:]  # after "..."
    out_sub = output[3:]  # after "..."

    # Should be 2 subscripts each
    if len(s1_sub) != 2 or len(s2_sub) != 2 or len(out_sub) != 2:
        return False

    # Pattern should be: s1=[...i,j], s2=[...j,k], out=[...i,k]
    # where j is the contracted dimension
    i1, j1 = s1_sub
    j2, k2 = s2_sub
    i_out, k_out = out_sub

    return j1 == j2 and i1 == i_out and k2 == k_out


def _is_sum_pattern(input_subscripts: List[str], output: Optional[str]) -> bool:
    """Check if the equation represents a sum/reduction operation."""
    if len(input_subscripts) != 1:
        return False

    s1 = input_subscripts[0]
    # All unique indices
    if len(set(s1)) != len(s1):
        return False

    if output is None or output == "":
        # Sum all elements
        return True
    elif set(output).issubset(set(s1)):
        # Sum over some dimensions
        return True
    return False


def _get_permutation(src: str, dst: str) -> List[int]:
    """Get permutation indices to transform src to dst."""
    return [src.index(c) for c in dst]


def _general_einsum_two_operands(
    equation: str, A: torch.Tensor, B: torch.Tensor
) -> torch.Tensor:
    """
    General einsum implementation for two operands using tensordot.

    This handles arbitrary contractions by:
    1. Identifying contracted dimensions
    2. Using torch.tensordot for the contraction
    3. Permuting the result to match the output specification
    """
    equation = equation.replace(" ", "")
    inputs_str, output = equation.split("->")
    s1, s2 = inputs_str.split(",")

    # Find contracted indices (appear in both inputs but not in output)
    set1 = set(s1)
    set2 = set(s2)
    set_out = set(output) if output else set()

    contracted = (set1 & set2) - set_out

    if not contracted:
        # No contraction - this is element-wise or outer product
        # Handle with broadcasting
        # Expand dimensions for proper broadcasting
        result = A.unsqueeze(-1) * B.unsqueeze(0)
        # Reshape and permute as needed
        all_indices = s1 + s2
        if output:
            perm = _get_permutation(all_indices, output)
            result = result.permute(perm)
        else:
            result = result.sum()
        return result.contiguous()

    # Find the positions of contracted indices in each operand
    dims_a = [s1.index(c) for c in contracted]
    dims_b = [s2.index(c) for c in contracted]

    # Use tensordot for contraction
    result = torch.tensordot(A, B, dims=(dims_a, dims_b))

    # Determine the result's index ordering after tensordot
    # tensordot puts non-contracted dims of A first, then non-contracted dims of B
    remaining_a = [c for c in s1 if c not in contracted]
    remaining_b = [c for c in s2 if c not in contracted]
    result_indices = remaining_a + remaining_b

    # Permute to match output if needed
    if output and result_indices != list(output):
        perm = _get_permutation("".join(result_indices), output)
        result = result.permute(perm)
    elif not output:
        # Sum all remaining dimensions
        result = result.sum()

    return result.contiguous()


def _general_einsum_one_operand(equation: str, A: torch.Tensor) -> torch.Tensor:
    """
    General einsum implementation for one operand.

    Handles general single-tensor operations by decomposing into:
    1. Diagonal extraction (for repeated indices)
    2. Summation (for indices not in output)
    3. Permutation (for reordering)
    """
    equation = equation.replace(" ", "")
    if "->" in equation:
        s1, output = equation.split("->")
    else:
        s1 = equation
        output = ""

    result = A

    # Handle repeated indices (diagonal extraction)
    # Find pairs of same indices
    from collections import Counter

    idx_counts = Counter(s1)
    repeated = [idx for idx, count in idx_counts.items() if count > 1]

    current_indices = list(s1)
    for idx in repeated:
        # Find positions of this repeated index
        positions = [i for i, c in enumerate(current_indices) if c == idx]
        if len(positions) >= 2:
            # Take diagonal along these dimensions
            dim1, dim2 = positions[0], positions[1]
            result = torch.diagonal(result, dim1=dim1, dim2=dim2)
            # After diagonal, those dimensions become the last dimension
            # Update current_indices
            new_indices = [
                c for i, c in enumerate(current_indices) if i not in [dim1, dim2]
            ] + [idx]
            current_indices = new_indices

    # Handle summation (indices not in output)
    unique_current = list(dict.fromkeys(current_indices))
    dims_to_reduce = [i for i, c in enumerate(unique_current) if c not in output]
    if dims_to_reduce:
        result = torch.sum(result, dim=dims_to_reduce, keepdim=False)
        remaining = [c for c in unique_current if c in output]
    else:
        remaining = unique_current

    # Handle permutation
    if output and remaining != list(output):
        perm = _get_permutation("".join(remaining), output)
        result = result.permute(perm)

    return result.contiguous() if result.dim() > 0 else result


def einsum(equation: str, *operands, path: Optional[List[int]] = None):
    """
    Einsum operation with optimized dispatch to FlagGems operations.

    Args:
        equation: Einstein summation equation string
        operands: Input tensors
        path: Optional contraction path (currently ignored, uses torch default)

    Returns:
        Result tensor
    """
    logger.debug("GEMS EINSUM")

    # Handle case where operands are passed as a list
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = tuple(operands[0])

    # Parse the equation
    input_subscripts, output, has_ellipsis = _parse_einsum_equation(equation)

    # For equations with ellipsis, try to handle common patterns
    if has_ellipsis:
        # Check for ellipsis batch matrix multiplication pattern
        if len(operands) == 2 and _is_ellipsis_bmm_pattern(equation):
            A, B = operands
            # Use torch.matmul which handles arbitrary batch dimensions
            return torch.matmul(A, B)
        # For other ellipsis patterns, reshape and compute
        # Fall through to general handling below

    # Try to match common patterns and dispatch to optimized implementations
    if len(operands) == 2:
        A, B = operands

        # Matrix multiplication: "ij,jk->ik"
        if _is_matmul_pattern(input_subscripts, output):
            return torch.mm(A, B)

        # Batch matrix multiplication: "bij,bjk->bik"
        if _is_bmm_pattern(input_subscripts, output):
            return torch.bmm(A, B)

        # Dot product: "i,i->"
        if _is_dot_pattern(input_subscripts, output):
            return torch.dot(A.flatten(), B.flatten())

        # Outer product: "i,j->ij"
        if _is_outer_pattern(input_subscripts, output):
            return torch.outer(A.flatten(), B.flatten())

    elif len(operands) == 1:
        A = operands[0]

        # Trace: "ii->"
        if _is_trace_pattern(input_subscripts, output):
            return torch.trace(A)

        # Diagonal: "ii->i"
        if _is_diagonal_pattern(input_subscripts, output):
            return torch.diagonal(A)

        # Transpose/permutation: "ijk->kji" etc.
        if _is_transpose_pattern(input_subscripts, output):
            perm = _get_permutation(input_subscripts[0], output)
            return A.permute(perm).contiguous()

        # Sum reduction: "ijk->" or "ijk->j"
        if _is_sum_pattern(input_subscripts, output):
            s1 = input_subscripts[0]
            if output is None or output == "":
                # Sum all elements
                return torch.sum(A)
            else:
                # Sum over specific dimensions
                dims_to_reduce = [i for i, c in enumerate(s1) if c not in output]
                result = torch.sum(A, dim=dims_to_reduce, keepdim=True)
                # Squeeze reduced dimensions and permute if needed
                for _ in dims_to_reduce:
                    result = result.squeeze()
                # Handle dimension permutation if output order differs
                remaining = [c for c in s1 if c in output]
                if remaining != list(output):
                    perm = _get_permutation("".join(remaining), output)
                    result = result.permute(perm)
                return result.contiguous()

    # For patterns not matching above, use general implementations
    operands = tuple(
        op.contiguous() if op.is_floating_point() else op for op in operands
    )

    if len(operands) == 2:
        # Use general two-operand implementation
        return _general_einsum_two_operands(equation, operands[0], operands[1])
    elif len(operands) == 1:
        # Use general one-operand implementation
        return _general_einsum_one_operand(equation, operands[0])
    else:
        # For 3+ operands, reduce pairwise
        # This is a simple left-to-right reduction (not optimal path)
        result = operands[0]
        for i in range(1, len(operands)):
            # This is a simplified approach; full implementation would
            # need proper equation splitting and intermediate subscript handling
            # For now, raise an error for complex cases
            raise NotImplementedError(
                f"einsum with {len(operands)} operands not yet supported. "
                "Consider breaking into multiple einsum calls."
            )
