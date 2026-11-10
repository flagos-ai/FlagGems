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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def histc_bin_count_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    min_val,
    max_val,
    bins,
    bins_per_prog,
    num_bin_progs,
    BLOCK_SIZE: tl.constexpr,
):
    """Atomic-free histogram.

    One program owns one (or more, when ``bins`` exceeds the grid limit)
    output bin and scans the whole input, counting the elements that map to
    that bin.  ``tt.atomic_rmw`` has no lowering on GCU300, so a scattering
    histogram is not an option; the extra reads are bandwidth-bound and
    acceptable for the bin counts used in practice.
    """
    pid = tl.program_id(0)

    for i in range(0, bins_per_prog):
        bin_id = pid + i * num_bin_progs
        if bin_id < bins:
            count = tl.zeros((), dtype=tl.int32)
            for start in range(0, n_elements, BLOCK_SIZE):
                offsets = start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                inp_val = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
                inp_val = inp_val.to(tl.float32)

                # Same binning formula as the CUDA kernel: elements equal to
                # max_val go to the last bin, values outside [min, max] and
                # NaN are ignored.
                bin_idx = tl.floor((inp_val - min_val) * bins / (max_val - min_val)).to(
                    tl.int32
                )
                bin_idx = tl.where(inp_val == max_val, bins - 1, bin_idx)
                bin_idx = tl.where(bin_idx < 0, 0, bin_idx)
                bin_idx = tl.where(bin_idx >= bins, bins - 1, bin_idx)

                # Combine comparisons as int masks: a vector boolean `&` of
                # float comparisons makes codegen emit an unrelocatable
                # vector `fabs` symbol on this target.
                ge_min = (inp_val >= min_val).to(tl.int32)
                le_max = (inp_val <= max_val).to(tl.int32)
                in_range = ge_min * le_max
                valid = in_range * mask.to(tl.int32)

                hit = (bin_idx == bin_id).to(tl.int32) * valid
                count += tl.sum(hit, axis=0)

            tl.store(out_ptr + bin_id, count.to(out_ptr.dtype.element_ty))


def histc(inp, bins=100, min=0, max=0):
    """
    Compute the histogram of a tensor.

    Args:
        inp: Input tensor
        bins: Number of histogram bins (default: 100)
        min: Lower end of the range (inclusive). If min == max == 0, uses data min.
        max: Upper end of the range (inclusive). If min == max == 0, uses data max.

    Returns:
        Tensor: Histogram represented as a tensor of shape (bins,)
    """
    logger.debug("GEMS ENFLAME HISTC")

    # Ensure input is contiguous
    inp = inp.contiguous()

    # Get min and max values
    min_val = float(min)
    max_val = float(max)

    if min_val == 0 and max_val == 0:
        # Use actual min/max of the data
        min_val = float(inp.min().item())
        max_val = float(inp.max().item())

    # Handle edge case where min == max
    if min_val == max_val:
        # All elements go to the first bin if they equal min_val
        out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)
        # Count how many elements equal min_val (excluding NaN)
        count = ((inp == min_val) & ~torch.isnan(inp)).sum().item()
        out[0] = count
        return out

    # Create output histogram tensor
    out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)

    n_elements = inp.numel()

    if n_elements == 0:
        return out

    # Choose block size
    BLOCK_SIZE = 1024

    # GCU300 caps grid.x at 65535: spread bins over the grid when needed.
    num_bin_progs = bins if bins < 65535 else 65535
    bins_per_prog = triton.cdiv(bins, num_bin_progs)

    with torch_device_fn.device(inp.device):
        histc_bin_count_kernel[(num_bin_progs,)](
            inp,
            out,
            n_elements,
            min_val,
            max_val,
            bins,
            bins_per_prog,
            num_bin_progs,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out
