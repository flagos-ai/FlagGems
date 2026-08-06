# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import torch

from flag_gems.ops.special_chebyshev_polynomial_u import (
    special_chebyshev_polynomial_u_kernel,
)

logger = logging.getLogger(__name__)


def special_chebyshev_polynomial_u(x, n):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_CHEBYSHEV_POLYNOMIAL_U")
    if isinstance(n, torch.Tensor):
        n = n.to(device=x.device, dtype=torch.int32)
        if torch.any((n < 0) | (n > 5)).item():
            raise ValueError("Chebyshev polynomial order n must be in [0, 5]")
    else:
        n_min = n_max = int(n)
        n = torch.empty((), dtype=torch.int32, device=x.device)
        n.fill_(n_min)
        if n_max > 5 or n_min < 0:
            raise ValueError(
                f"Chebyshev polynomial order n must be in [0, 5], "
                f"got values in [{n_min}, {n_max}]"
            )

    return special_chebyshev_polynomial_u_kernel(x, n)
