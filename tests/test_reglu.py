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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "reglu", None)
except ImportError:
    TE_OP = None
except RuntimeError as te_import_error:
    # The installed transformer-engine here is only the PyPI metapackage, so
    # transformer_engine.common aborts while importing and turns the whole
    # module uncollectable.  The TE reference op has no GCU implementation
    # (torch aten/TE not supported on this backend), so skip the module on
    # enflame with the real cause instead of failing collection for every
    # operator of the run.
    if flag_gems.vendor_name == "enflame":
        pytest.skip(
            "enflame GCU: transformer-engine metapackage has no framework "
            f"extension ({te_import_error}); the TE reference op is not "
            "supported on this backend (torch aten/TE not supported on GCU)",
            allow_module_level=True,
        )
    raise


def _reference_reglu(input_tensor: torch.Tensor, quantizer=None) -> torch.Tensor:
    """High-precision (fp64) reference for ReGLU.

    ``ReGLU(a, b)`` with ``a = input[..., :N]``, ``b = input[..., N:]`` is
    ``relu(a) * b``. Takes the same arguments as TransformerEngine's entry point
    so it can stand in for it when TE is not installed.
    """
    n = input_tensor.shape[-1] // 2
    a = input_tensor[..., :n].to(torch.float64)
    b = input_tensor[..., n:].to(torch.float64)
    return torch.clamp(a, min=0.0) * b


if TE_OP is None:
    # TransformerEngine is not installed: bind the fp64 torch reference in
    # place of the TE entry point so the accuracy test runs instead of skipping.
    TE_OP = _reference_reglu


@pytest.mark.reglu
@pytest.mark.parametrize("shape", utils.GLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reglu(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = TE_OP(input_tensor, None)
    ref_out = utils.to_reference(ref_out)
    res_out = flag_gems.reglu(input_tensor)

    utils.gems_assert_close(res_out, ref_out, dtype)
