import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_acos = tl_extra_shim.acos
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def acos_kernel(x):
        """
            Compute arccos(x) for input x.

                    The arccos function returns values in the range [0, π] for input values in [-1, 1].

                            Args:
                                    x: Input tensor (will be converted to float32 for computation)

                                                Returns:
                                                        Tensor with arccos(x) computed element-wise
                                                            """
        return _acos(x.to(tl.float32))


def acos(x):
        """
            Computes the inverse cosine (arccos) of each element in input.

                    Args:
                            x (Tensor): Input tensor with values in [-1, 1]

                                        Returns:
                                                Tensor: Output tensor with values in [0, π]

                                                            Example:
                                                                    >>> x = torch.tensor([0.0, 0.5, 1.0])
                                                                            >>> torch.acos(x)
                                                                                    tensor([1.5708, 1.0472, 0.0000])
                                                                                        """
        logger.debug("GEMS ACOS FORWARD")
        y = acos_kernel(x)
        return y


def acos_(x):
        """
            In-place version of acos.

                    Computes the inverse cosine of each element in input, modifying the tensor in-place.

                            Args:
                                    x (Tensor): Input tensor with values in [-1, 1] (modified in-place)

                                                Returns:
                                                        Tensor: The modified input tensor

                                                                    Example:
                                                                            >>> x = torch.tensor([0.0, 0.5, 1.0])
                                                                                    >>> torch.acos_(x)
                                                                                            tensor([1.5708, 1.0472, 0.0000])
                                                                                                """
        logger.debug("GEMS ACOS_ INPLACE")
        acos_kernel(x, out0=x)
        return x
