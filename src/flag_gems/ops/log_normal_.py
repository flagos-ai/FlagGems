import logging

from flag_gems.ops.normal import normal_distribution

logger = logging.getLogger(__name__)


def log_normal_(self, mean=1, std=2, *, generator=None):
    logger.debug("GEMS LOG_NORMAL_")
    normal_distribution(self.shape, self.device, generator=generator, out=self)
    self.mul_(std).add_(mean).exp_()
    return self
