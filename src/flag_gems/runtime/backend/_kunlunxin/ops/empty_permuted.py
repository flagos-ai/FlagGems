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
#
import logging

import torch

logger = logging.getLogger("flag_gems.ops.empty_permuted")


def _strides_from_physical_layout(size, physical_layout):
    """Derives strides from a physical layout.

    `physical_layout` lists the logical dimensions ordered from the outermost to
    the innermost physical dimension, so walking it backwards and accumulating
    the sizes yields the stride of each logical dimension.
    """
    strides = [0] * len(size)
    accumulated = 1
    for dim in reversed(physical_layout):
        strides[dim] = accumulated
        accumulated *= size[dim]
    return strides


def empty_permuted(
    size,
    physical_layout,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    """Returns an uninitialized tensor whose memory follows `physical_layout`."""
    logger.debug("GEMS EMPTY_PERMUTED")
    shape = tuple(size)
    physical_layout = list(physical_layout)
    if len(physical_layout) != len(shape):
        raise RuntimeError(
            f"empty_permuted: physical_layout must have the same length as size, "
            f"but got {len(physical_layout)} and {len(shape)}"
        )
    if sorted(physical_layout) != list(range(len(shape))):
        raise RuntimeError(
            f"empty_permuted: physical_layout must be a permutation of "
            f"[0, {len(shape)}), but got {physical_layout}"
        )

    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        import flag_gems.runtime as _rt

        device = torch.device(_rt.device.name)
    if layout is None:
        layout = torch.strided
    if pin_memory is None:
        pin_memory = False

    # Allocation only — uninitialized by definition; nothing is written. The
    # native `empty_permuted` does exactly the same, so the override matches
    # the vendor allocation engine without an extra kernel launch.
    out = torch.empty_strided(
        shape,
        _strides_from_physical_layout(shape, physical_layout),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    return out
