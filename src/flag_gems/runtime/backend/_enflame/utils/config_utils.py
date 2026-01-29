import torch
import importlib.util
if importlib.util.find_spec("triton.backends.enflame") is None:
    from triton_gcu.triton.driver import _GCUDriver
else:
    from triton.backends.enflame.driver import _GCUDriver

import re

driver = _GCUDriver()
arch = driver.get_arch()
arch_version = int(re.search(r'gcu(\d+)', arch).group(1))

if arch_version == 300:
    MAX_GRID_DIM = 128
elif arch_version >= 400:
    MAX_GRID_DIM = 48