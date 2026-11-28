import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SHAPE_DEPTHWISE = [
    ((2, 4, 8, 8, 8), (4, 1, 2, 2, 2), (2, 2, 2)),
    ((3, 16, 6, 6, 6), (16, 1, 2, 2, 2), (2, 2, 2)),
    ((2, 8, 5, 7, 9), (8, 1, 3, 3, 3), (3, 3, 3)),
]

# torch_gcu provides no PrivateUse1 kernel for aten::conv_depthwise3d, and the
# operator has no CPU implementation at all (see the comment on the reference
# call below), so on that backend there is no way to build the golden value:
# the PrivateUse1 fallback lands on a missing CPU kernel and raises. Skip the
# module there rather than comparing against a different operator.
ATEN_REFERENCE_UNSUPPORTED = flag_gems.vendor_name == "enflame"


@pytest.mark.skipif(
    ATEN_REFERENCE_UNSUPPORTED,
    reason="aten::conv_depthwise3d has no PrivateUse1 nor CPU kernel on this backend",
)
@pytest.mark.conv_depthwise3d
@pytest.mark.parametrize("shape_input, shape_weight, kernel", SHAPE_DEPTHWISE)
@pytest.mark.parametrize("stride", [[1, 1, 1], [2, 2, 2]])
@pytest.mark.parametrize("padding", [[0, 0, 0], [1, 1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1, 1]])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("bias", [True, False])
def test_conv_depthwise3d(
    shape_input, shape_weight, kernel, stride, padding, dilation, dtype, bias
):
    inp = torch.randn(shape_input, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(shape_weight, dtype=dtype, device=flag_gems.device)

    if bias:
        bias_tensor = torch.randn(shape_weight[0], dtype=dtype, device=flag_gems.device)
    else:
        bias_tensor = None

    torch.backends.cudnn.allow_tf32 = False
    # aten::conv_depthwise3d has no CPU implementation, so the reference has to
    # run on the device even under --ref cpu; gems_assert_close moves both
    # tensors to the CPU before comparing.
    ref_out = torch.ops.aten.conv_depthwise3d(
        inp,
        weight,
        kernel,
        bias_tensor,
        stride,
        padding,
        dilation,
    )

    res_out = flag_gems.conv_depthwise3d(
        inp, weight, kernel, bias_tensor, stride, padding, dilation
    )
    utils.gems_assert_close(res_out, ref_out.cpu(), dtype)
