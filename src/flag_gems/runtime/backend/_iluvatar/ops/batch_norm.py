import logging

from torch import Tensor

from flag_gems import ops

logger = logging.getLogger(__name__)


def batch_norm(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("ILUVATAR GEMS BATCHNORM")
    return ops.batch_norm(
        input,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=training,
        momentum=momentum,
        eps=eps,
    )


def batch_norm_backward(
    grad_out,
    input,
    weight=None,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_invstd=None,
    train=False,
    eps=1e-05,
    output_mask=None,
):
    logger.debug("ILUVATAR GEMS BATCHNORM BACKWARD")
    return ops.batch_norm_backward(
        grad_out,
        input,
        weight=weight,
        running_mean=running_mean,
        running_var=running_var,
        save_mean=save_mean,
        save_invstd=save_invstd,
        train=train,
        eps=eps,
        output_mask=output_mask,
    )
