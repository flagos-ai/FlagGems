import importlib.util
import os
import re

from backend_utils import VendorInfoBase

# NOTE: transfer_to_gcu is not used anywhere
# try:
#     from torch_gcu import transfer_to_gcu  # noqa: F401
# except Exception:
#    logger.warning("torch_gcu not installed")

# TODO: Revise the following imports to be exception free
if importlib.util.find_spec("triton.backends.enflame") is None:
    from triton_gcu.triton.driver import _GCUDriver
else:
    from triton.backends.enflame.driver import _GCUDriver

driver = _GCUDriver()
arch = driver.get_arch()
arch_version = int(re.search(r"gcu(\d+)", arch).group(1))

vendor_info = VendorInfoBase(
    vendor_name="enflame",
    device_name="gcu",
    device_query_cmd="",
    dispatch_key="PrivateUse1",
)

os.environ["ARCH"] = str(arch_version)
ARCH_MAP = {"3": "gcu300", "4": "gcu400"}
# i64 to/copy is not supported in gcu300
CUSTOMIZED_UNUSED_OPS = (
    "to_copy",
    "copy_",
)

__all__ = ["*"]
