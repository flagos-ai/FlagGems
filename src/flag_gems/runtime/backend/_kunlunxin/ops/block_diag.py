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

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def block_diag(*tensors):
    """Block diagonal matrix construction (Kunlunxin override)."""
    logger.debug("GEMS_KUNLUNXIN BLOCK_DIAG")

    # Handle case where tensors is passed as a single list/tuple
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])

    if len(tensors) == 0:
        return torch.empty((1, 0))

    # Normalize to 2D: 0D -> (1, 1), 1D -> (1, K), 2D as-is.
    tensors_2d = []
    for t in tensors:
        if t.ndim == 0:
            tensors_2d.append(t.unsqueeze(0).unsqueeze(0))
        elif t.ndim == 1:
            tensors_2d.append(t.unsqueeze(0))
        else:
            assert t.ndim == 2, f"Expected 0D, 1D, or 2D tensor, got {t.ndim}D"
            tensors_2d.append(t)

    total_rows = sum(t.shape[0] for t in tensors_2d)
    total_cols = sum(t.shape[1] for t in tensors_2d)

    out_dtype = tensors_2d[0].dtype
    for t in tensors_2d[1:]:
        out_dtype = torch.promote_types(out_dtype, t.dtype)
    device = tensors_2d[0].device

    out = torch.zeros((total_rows, total_cols), dtype=out_dtype, device=device)

    row_off = 0
    col_off = 0
    for t in tensors_2d:
        rows, cols = t.shape
        if rows > 0 and cols > 0:
            src = (
                t
                if (t.is_contiguous() and t.dtype == out_dtype)
                else t.contiguous().to(out_dtype)
            )
            out[row_off : row_off + rows, col_off : col_off + cols] = src
        row_off += rows
        col_off += cols

    return out
