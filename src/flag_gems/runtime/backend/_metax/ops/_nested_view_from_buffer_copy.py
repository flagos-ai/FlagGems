import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def _nested_view_from_buffer_copy(
    self: torch.Tensor,
    nested_size: torch.Tensor,
    nested_strides: torch.Tensor,
    offsets: torch.Tensor,
):
    logger.debug("GEMS_METAX _NESTED_VIEW_FROM_BUFFER_COPY")

    # Metax PyTorch has a segfault issue with _nested_view_from_buffer_copy on CUDA
    # Workaround: manually extract components and create nested tensor
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

        # Extract component using as_strided
        # The buffer may have different layout, so we use as_strided
        component = self.as_strided(
            (size_i,), (stride_i,), offset_i
        ).clone()  # Clone to ensure contiguous memory

        components.append(component)

    # Create nested tensor using torch.nested.nested_tensor
    result = torch.nested.nested_tensor(components)
    return result
