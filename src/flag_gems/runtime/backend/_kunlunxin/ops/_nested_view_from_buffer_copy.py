import logging

import torch

from ..utils.tle_copy import tle_copy

logger = logging.getLogger("flag_gems." + __name__)


def _nested_view_from_buffer_copy(
    self: torch.Tensor,
    nested_size: torch.Tensor,
    nested_strides: torch.Tensor,
    offsets: torch.Tensor,
):
    logger.debug("GEMS_KUNLUNXIN _NESTED_VIEW_FROM_BUFFER_COPY")
    num_components = nested_size.shape[0]

    if (
        self.dim() == 1
        and nested_size.dim() == 2
        and nested_size.shape[1] == 1
        and nested_size.dtype == torch.int64
        and nested_strides.dtype == torch.int64
        and offsets.dtype == torch.int64
        and all(s == 1 for s in nested_strides.reshape(-1).tolist())
    ):
        values = torch.empty_strided(
            self.shape, self.stride(), dtype=self.dtype, device=self.device
        )
        if not tle_copy(self, values):
            torch.ops.aten._copy_from(self, values, False)
        full_offsets = torch.empty_strided(
            (num_components + 1,), (1,), dtype=torch.int64, device=self.device
        )
        if not tle_copy(offsets, full_offsets[:num_components]):
            torch.ops.aten._copy_from(offsets, full_offsets[:num_components], False)
        if not tle_copy(offsets[:1], full_offsets[num_components:]):
            torch.ops.aten._copy_from(offsets[:1], full_offsets[num_components:], False)
        from torch.nested._internal.nested_tensor import NestedTensor

        return NestedTensor(
            values,
            full_offsets,
            lengths=nested_size[:, 0],
            _ragged_idx=1,
        )

    snapshot = torch.empty_strided(
        self.shape, self.stride(), dtype=self.dtype, device=self.device
    )
    if not tle_copy(self, snapshot):
        torch.ops.aten._copy_from(self, snapshot, False)

    num_components = nested_size.shape[0]
    components = []
    for i in range(num_components):
        size_i = int(nested_size[i].item())
        stride_i = (
            int(nested_strides[i].item())
            if nested_strides.ndim > 1
            else int(nested_strides[i].item())
        )
        offset_i = int(offsets[i].item())
        components.append(snapshot.as_strided((size_i,), (stride_i,), offset_i))

    return torch.nested.as_nested_tensor(components)


__all__ = ["_nested_view_from_buffer_copy"]
