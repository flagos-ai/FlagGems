import logging
from pathlib import Path


class LogOncePerLocationFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.logged_locations = set()

    def filter(self, record):
        key = (record.pathname, record.lineno)
        if key in self.logged_locations:
            return False
        self.logged_locations.add(key)
        return True


def setup_flaggems_logging(path=None, record=True, once=False):
    if not record:
        return

    filename = Path(path or Path.home() / ".flaggems/oplist.log")
    handler = logging.FileHandler(filename, mode="w")

    if once:
        handler.addFilter(LogOncePerLocationFilter())

    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(funcName)s %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger("flag_gems")
    logger.setLevel(logging.DEBUG)

    import builtins

    if not builtins.any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(handler)

    logger.propagate = False
