# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def histogram_kernel(
    input_ptr,
    weight_ptr,
    hist_ptr,
    n_elements,
    num_bins: tl.constexpr,
    min_val,
    max_val,
    has_weight: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to compute histogram with atomic adds (unused placeholder)."""
    # This kernel is currently unused - we use bucketize for correctness
    # It's kept as a placeholder for future optimization
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Compute bin width
    bin_width = (max_val - min_val) / num_bins

    # Process each element with atomic adds
    # We need to iterate because each element may go to a different bin
    for i in range(BLOCK_SIZE):
        offset = block_start + i
        if offset < n_elements:
            val = tl.load(input_ptr + offset)

            if has_weight:
                weight = tl.load(weight_ptr + offset)
            else:
                weight = 1.0

            # Compute bin index
            # Skip if out of range or NaN
            valid = (val >= min_val) & (val == val)  # val == val checks for non-NaN
            if valid:
                if val >= max_val:
                    bin_idx = num_bins - 1
                else:
                    # Compute bin index with integer division
                    bin_idx_float = (val - min_val) / bin_width
                    bin_idx = bin_idx_float.to(tl.int32)
                    # Clamp to valid range
                    bin_idx = tl.minimum(tl.maximum(bin_idx, 0), num_bins - 1)

                # Atomic add to histogram
                tl.atomic_add(hist_ptr + bin_idx, weight)


def histogram_bin_ct(self, bins=100, *, range=None, weight=None, density=False):
    """
    Compute histogram with uniform bins (integer bin count).

    Args:
        self: input tensor (any shape)
        bins: number of bins (int, default 100)
        range: tuple of (min, max) or None (default None = use data range)
        weight: optional weight tensor (same shape as self) or None
        density: if True, normalize to form probability density

    Returns:
        hist: histogram counts (shape: [bins])
        bin_edges: bin edges (shape: [bins+1])
    """
    logger.debug("GEMS HISTOGRAM.BIN_CT")

    # Flatten input
    input_flat = self.flatten().contiguous()
    n_elements = input_flat.numel()

    # Determine range
    if range is None:
        min_val = float(input_flat.min().item())
        max_val = float(input_flat.max().item())
    else:
        min_val, max_val = float(range[0]), float(range[1])

    # Handle edge case where min == max
    if min_val == max_val:
        min_val = min_val - 0.5
        max_val = max_val + 0.5

    # Create bin edges (match torch output dtype)
    bin_edges = torch.linspace(
        min_val, max_val, bins + 1, dtype=self.dtype, device=self.device
    )

    # Allocate histogram (match torch output dtype)
    hist = torch.zeros(bins, dtype=self.dtype, device=self.device)

    if n_elements == 0:
        return hist, bin_edges

    # Use bucketize for correct bin assignment (matches torch CPU)
    # bucketize returns indices in [0, bins], subtract 1 for [−1, bins-1]
    indices = torch.bucketize(input_flat, bin_edges, right=True) - 1

    # Elements exactly equal to the last edge go to the last bin
    indices = torch.where(
        input_flat == bin_edges[-1],
        torch.full_like(indices, bins - 1),
        indices,
    )

    # Filter valid indices (in range and not NaN)
    valid_mask = (indices >= 0) & (indices < bins)
    indices = indices[valid_mask]

    # Handle weights
    if weight is not None:
        weight_flat = weight.flatten().contiguous()[valid_mask]
        hist.scatter_add_(0, indices.long(), weight_flat)
    else:
        hist.scatter_add_(0, indices.long(), torch.ones_like(indices, dtype=hist.dtype))

    # Apply density normalization if requested
    if density:
        bin_width = (max_val - min_val) / bins
        total = hist.sum()
        if total > 0:
            hist = hist / (total * bin_width)

    return hist, bin_edges


def histogram_bins_tensor(self, bins, *, weight=None, density=False):
    """
    Compute histogram with non-uniform bins (tensor bin edges).

    Args:
        self: input tensor (any shape)
        bins: 1D tensor of bin edges (must be monotonically increasing)
        weight: optional weight tensor (same shape as self) or None
        density: if True, normalize to form probability density

    Returns:
        hist: histogram counts (shape: [bins.numel()-1])
        bin_edges: bin edges (copy of bins)
    """
    logger.debug("GEMS HISTOGRAM.BINS_TENSOR")

    bins = bins.contiguous()
    num_bins = bins.numel() - 1

    if num_bins <= 0:
        raise ValueError("bins must have at least 2 elements")

    # For non-uniform bins, use torch.bucketize + bincount
    # This is simpler and correct for non-uniform case
    input_flat = self.flatten()
    n_elements = input_flat.numel()

    # Allocate histogram
    hist = torch.zeros(num_bins, dtype=self.dtype, device=self.device)

    if n_elements == 0:
        return hist, bins.clone()

    # Use bucketize to find bin indices (right=True for histogram)
    # bucketize returns indices in [0, bins], subtract 1 for [−1, bins-1]
    indices = torch.bucketize(input_flat, bins, right=True) - 1

    # Handle edge case: values exactly equal to last edge go to last bin
    indices = torch.where(
        input_flat == bins[-1],
        torch.full_like(indices, num_bins - 1),
        indices,
    )

    # Clamp to valid range and filter out-of-range values
    valid_mask = (indices >= 0) & (indices < num_bins)
    indices = indices[valid_mask]

    # Handle weights
    if weight is not None:
        weight_flat = weight.flatten()[valid_mask]
        hist.scatter_add_(0, indices.long(), weight_flat)
    else:
        hist.scatter_add_(0, indices.long(), torch.ones_like(indices, dtype=hist.dtype))

    # Apply density normalization if requested
    if density:
        bin_widths = bins[1:] - bins[:-1]
        total = hist.sum()
        if total > 0:
            hist = hist / (total * bin_widths)

    return hist, bins.clone()
