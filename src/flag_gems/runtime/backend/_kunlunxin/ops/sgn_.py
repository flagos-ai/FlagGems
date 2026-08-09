# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

from .sgn import _sgn_impl

logger = logging.getLogger(__name__)


def sgn_(x):
    logger.debug("GEMS_KUNLUNXIN SGN_")
    return _sgn_impl(x, x)
