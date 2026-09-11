import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def _nested_view_from_buffer_copy(
    self: torch.Tensor,
    nested_size: torch.Tensor,
    nested_strides: torch.Tensor,
    offsets: torch.Tensor,
):
    logger.debug("GEMS CAMBRICON _NESTED_VIEW_FROM_BUFFER_COPY")

    # The Cambricon (mlu) NestedTensor backend does not implement several ops
    # (e.g. aten::unbind.int, aten::_to_copy), so a nested tensor built directly
    # on the device is unusable and torch.nested.nested_tensor on device tensors
    # also fails. In addition, torch.nested.nested_tensor under use_gems would
    # dispatch through the Cambricon cat/vstack path.
    #
    # Workaround: extract each component on device using as_strided, move the
    # (small, contiguous) components to CPU, and assemble the nested tensor on
    # CPU where unbind and the rest of the nested API are fully supported.
    num_components = nested_size.shape[0]
    components = []

    for i in range(num_components):
        size_i = int(nested_size[i].item())
        stride_i = int(nested_strides[i].item())
        offset_i = int(offsets[i].item())

        # Extract component and clone to obtain contiguous memory before the
        # host copy.
        component = self.as_strided((size_i,), (stride_i,), offset_i).clone().cpu()
        components.append(component)

    result = torch.nested.nested_tensor(components)
    return result
