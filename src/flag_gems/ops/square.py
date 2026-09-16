# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def square_func(x):
    return x * x


def square(A):
    logger.debug("GEMS SQUARE")
    return square_func(A)


def square_out(A, *, out=None):
    logger.debug("GEMS SQUARE_OUT")
    if out is None:
        return square_func(A)
    square_func(A, out0=out)
    return out


def square_(A):
    logger.debug("GEMS SQUARE_")
    square_func(A, out0=A)
    return A
