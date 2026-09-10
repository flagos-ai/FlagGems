# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

from .native_batch_norm import native_batch_norm

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

    # NOTE (kunlunxin / XPU, 2026-09-02): delegate to the vendor
    # `native_batch_norm` kernels instead of `batch_norm`, because
    # `tests/test_native_batch_norm_legit_functional.py` compares against the CPU
    # `aten::_native_batch_norm_legit_functional` reference, which:
    #   * folds the UNBIASED batch variance (var * count / (count - 1)) into
    #     running_var, while the vendor `batch_norm` used the biased one (fails
    #     for small N*S shapes, e.g. (16, 3): n = 16, +6.7% error);
    #   * updates the running stats for float16/bfloat16 too, while the vendor
    #     `batch_norm` restricted the in-place update to float32 (fp16/bf16
    #     running stats were left at their initial values, 100% mismatch).
    # The vendor `native_batch_norm` implements exactly those two semantics
    # (validated by tests/test_batch_norm.py::test_native_batch_norm against the
    # CPU reference) and updates running_mean / running_var in place, which is
    # what the functional variant returns in its 4th/5th tuple slots.
    output, save_mean, save_invstd = native_batch_norm(
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
