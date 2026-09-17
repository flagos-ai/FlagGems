import logging

from .round import round as _round
from .round import round_out as _round_out

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def special_round(input, *, decimals=0):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ROUND")
    # aten::special_round may pass decimals=None when the caller omits it;
    # the schema default is 0, normalize so the decimals==0 fast path is taken.
    return _round(input, decimals=0 if decimals is None else decimals)


def special_round_out(input, out, *, decimals=0):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ROUND_OUT")
    return _round_out(input, decimals=0 if decimals is None else decimals, out=out)
