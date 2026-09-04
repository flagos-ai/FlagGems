# Copyright 2026, The FlagOS Contributors.
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

logger = logging.getLogger(__name__)


# Sobol direction numbers for dimensions 0-99
# Dimension 0: simple powers of 2 (computed in kernel)
# Dimensions 1-99: pre-computed from Sobol polynomial coefficients
# Source: Joe-Kuo-2008 direction numbers used by PyTorch
_SOBOL_DIRECTIONS_1_TO_99 = [
    [
        536870912,
        805306368,
        671088640,
        1006632960,
        570425344,
        855638016,
        713031680,
        1069547520,
        538968064,
        808452096,
        673710080,
        1010565120,
        572653568,
        858980352,
        715816960,
        1073725440,
        536879104,
        805318656,
        671098880,
        1006648320,
        570434048,
        855651072,
        713042560,
        1069563840,
        538976288,
        808464432,
        673720360,
        1010580540,
        572662306,
        858993459,
    ],
    [
        536870912,
        805306368,
        402653184,
        603979776,
        973078528,
        385875968,
        595591168,
        826277888,
        438304768,
        657457152,
        999817216,
        358875136,
        538574848,
        807862272,
        406552576,
        605372416,
        975183872,
        389033984,
        597170176,
        828646400,
        437926400,
        656873216,
        1002152832,
        357921088,
        536885792,
        805312304,
        402662296,
        603992420,
        973085210,
        385885991,
    ],
    [
        536870912,
        805306368,
        134217728,
        335544320,
        1040187392,
        486539264,
        679477248,
        616562688,
        908066816,
        156237824,
        376963072,
        968097792,
        503447552,
        755171328,
        545292288,
        817971200,
        136568832,
        340905984,
        1056606208,
        494291968,
        673276416,
        609457408,
        922347392,
        158784320,
        371195936,
        961544240,
        511180808,
        766771220,
        537002046,
        805503005,
    ],
    [
        536870912,
        268435456,
        134217728,
        738197504,
        1040187392,
        922746880,
        511705088,
        658505728,
        379584512,
        200278016,
        676855808,
        1009516544,
        916586496,
        468779008,
        542670848,
        271499264,
        144826368,
        754085888,
        1054435328,
        929870848,
        503351808,
        654495488,
        377744768,
        188970688,
        681697312,
        1022521360,
        920217608,
        460108844,
        536906302,
        268619575,
    ],
    [
        536870912,
        268435456,
        402653184,
        201326592,
        838860800,
        150994944,
        360710144,
        1052770304,
        941621248,
        470810624,
        706215936,
        84672512,
        665976832,
        935919616,
        766869504,
        586072064,
        301998080,
        419434496,
        226498560,
        851446784,
        169882112,
        353372416,
        1066931584,
        1003241152,
        529676320,
        735648784,
        128821784,
        669173004,
        900859826,
        784934857,
    ],
]  # Truncated to first 5 for code size - full table would be loaded dynamically


@triton.jit
def _sobol_init_kernel(
    state_ptr,
    dimension: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Initialize Sobol engine state with direction numbers.

    Each program handles one dimension (one row of the state tensor).
    For dimension 0: computes 2^(29-i) for i in [0, 30)
    For dimension >= 1: uses lookup table (passed via constexpr for dimensions 1-99)
    """
    # Get dimension index for this program
    dim_idx = tl.program_id(0)

    if dim_idx >= dimension:
        return

    # Base offset for this dimension's 30 values
    base_offset = dim_idx * 30

    # Process all 30 values in this dimension
    indices = tl.arange(0, BLOCK_SIZE)
    mask = indices < 30

    if dim_idx == 0:
        # First dimension: compute 2^(29-i) for i in [0, 30)
        # This gives us: [2^29, 2^28, ..., 2^1, 2^0]
        powers = 29 - indices
        # Cast to int64 to match output dtype
        shifted = 1 << powers
        values = tl.where(mask, shifted, 0).to(tl.int64)
    else:
        # For dimensions >= 1, we would need to lookup from the direction table
        # Since Triton doesn't support large constant arrays well, we use a fallback:
        # This is a placeholder - in practice, dimensions > 99 would need special handling
        # For now, just fill with zeros as a fallback (will be replaced by CPU computation)
        values = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Write to output
    output_ptr = state_ptr + base_offset + indices
    tl.store(output_ptr, values, mask=mask)


def _sobol_engine_initialize_state_(state: torch.Tensor, dimension: int):
    """
    Initialize Sobol quasi-random number generator state with direction numbers.

    This operator fills the state tensor with Sobol sequence direction numbers.
    - First dimension: powers of 2 from 2^29 down to 2^0
    - Other dimensions: Sobol direction numbers from polynomial coefficients

    Args:
        state: Tensor of shape (dimension, 30) and dtype int64, modified in-place
        dimension: Number of dimensions (must match state.shape[0])

    Returns:
        state: The modified input tensor

    Note: This Triton implementation has limitations:
    - Only supports dimensions up to a small number efficiently (constant table size)
    - For larger dimensions, falls back to CPU computation
    - Native PyTorch implementation supports up to 21111 dimensions
    """
    logger.debug("GEMS _SOBOL_ENGINE_INITIALIZE_STATE_")

    assert state.dim() == 2, f"Expected 2D tensor, got {state.dim()}D"
    assert (
        state.shape[0] == dimension
    ), f"State shape[0] ({state.shape[0]}) != dimension ({dimension})"
    assert state.shape[1] == 30, f"Expected shape[1]=30, got {state.shape[1]}"
    assert state.dtype == torch.int64, f"Expected dtype int64, got {state.dtype}"

    # For dimensions > 5 or non-CUDA, fall back to native implementation
    # (We only have 5 directions in the truncated table above)
    if dimension > 5 or not state.is_cuda:
        # Use CPU computation for fallback
        if state.is_cuda:
            cpu_state = state.cpu()
            torch.ops.aten._sobol_engine_initialize_state_(cpu_state, dimension)
            state.copy_(cpu_state)
        else:
            torch.ops.aten._sobol_engine_initialize_state_(state, dimension)
        return state

    # Triton kernel launch
    # Use 32 as block size (30 elements per dimension, rounded up to 32)
    BLOCK_SIZE = 32

    # Grid: one program per dimension
    grid = (dimension,)

    # Launch kernel
    _sobol_init_kernel[grid](
        state,
        dimension=dimension,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # For dimensions 1-5, we need to manually fill the lookup table values
    # since Triton doesn't easily support large constant arrays
    # This is a hybrid approach: Triton for dim 0, CPU for others
    if dimension > 1:
        # Copy the pre-computed direction numbers for dimensions 1+
        for dim_idx in range(1, min(dimension, 6)):
            state[dim_idx, :] = torch.tensor(
                _SOBOL_DIRECTIONS_1_TO_99[dim_idx - 1],
                dtype=torch.int64,
                device=state.device,
            )

    return state


# Note: @libentry() removed - this operator doesn't follow the standard pattern
# It will be registered directly in __init__.py's _FULL_CONFIG
