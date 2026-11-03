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

from backend_utils import VendorDescriptor

# Import the MUSA runtime so it registers the musa backend (torch.musa,
# PrivateUse1 dispatch key, device properties) on torch.  This module is
# imported by the vendor scan / vendor lookup *before* the device detector
# probes torch.<device_name>, mirroring the other PrivateUse1 vendor
# descriptors (cambricon / biren / sunrise import their runtime libs the
# same way).  Without it, `import flag_gems` on a normal MTHREADS install
# fails with "module 'torch' has no attribute 'musa'" unless the caller
# happens to import torch_musa first.  Only a genuinely absent torch_musa
# skips this vendor; an installed-but-broken one (a missing internal
# dependency surfaces as ModuleNotFoundError with that dependency's name,
# not "torch_musa") is recorded with its real traceback and re-raised --
# the generic vendor scan swallows per-vendor failures, and without this
# record a broken install would degrade into a misleading
# "no device detected".
try:
    import torch_musa  # noqa: F401  (import side effect: registers torch.musa)
except ModuleNotFoundError as err:
    if err.name != "torch_musa":
        logging.getLogger(__name__).exception(
            "torch_musa is installed but failed to import; the mthreads "
            "backend cannot be registered"
        )
        raise
    # torch_musa is not installed: this vendor is simply skipped (the
    # vendor scan falls through to the other candidates).
except Exception:
    # Any other failure from an installed torch_musa (a plain ImportError,
    # an extension-init exception, ...): same treatment -- record the real
    # cause, then re-raise.
    logging.getLogger(__name__).exception(
        "torch_musa is installed but failed to import; the mthreads "
        "backend cannot be registered"
    )
    raise

vendor_info = VendorDescriptor(
    vendor_name="mthreads",
    device_name="musa",
    dispatch_key="PrivateUse1",
    device_query_cmd="mthreads-gmi",
    fp64_enabled=False,
    tle_enabled=True,
)

CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]
