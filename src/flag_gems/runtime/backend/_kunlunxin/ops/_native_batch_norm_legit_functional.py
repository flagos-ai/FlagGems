# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

from .batch_norm import batch_norm

logger = logging.getLogger(__name__)


def _native_batch_norm_legit_functional(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_FUNCTIONAL")
    output, save_mean, save_invstd = batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    )
    return output, save_mean, save_invstd, running_mean, running_var
