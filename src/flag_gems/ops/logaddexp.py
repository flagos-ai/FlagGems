import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp_func(a, b):
    a32 = a.to(tl.float32)
    b32 = b.to(tl.float32)

    # NaN detection (NaN != NaN by IEEE 754)
    a_is_nan = a32 != a32
    b_is_nan = b32 != b32
    has_nan = a_is_nan | b_is_nan

    max_val = tl.maximum(a32, b32)
    min_val = tl.minimum(a32, b32)

    # Standard numerically stable formula
    # When diff is very negative, exp(diff) ≈ 0, so result ≈ max_val
    diff = min_val - max_val
    result = max_val + tl.log(1.0 + tl.exp(diff))

    # Fix special cases using cascaded where() - order matters!
    # Note: In Triton, all branches are evaluated, so we're just selecting results

    # 1. If max == -inf, both must be -inf (since max is the larger value)
    #    Formula gives: -inf + log(1 + exp(nan)) = nan, but we want -inf
    result = tl.where(max_val == float("-inf"), float("-inf"), result)

    # 2. If max == +inf (and we get here), return +inf
    #    This handles both logaddexp(inf, inf) and logaddexp(inf, finite)
    result = tl.where(max_val == float("inf"), float("inf"), result)

    # 3. NaN propagation (must be last to override everything)
    result = tl.where(has_nan, float("nan"), result)

    return result


def logaddexp(A, B):
    logger.debug("GEMS LOGADDEXP")
    return logaddexp_func(A, B)
