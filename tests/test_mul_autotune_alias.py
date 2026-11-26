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

"""Cold autotuning must preserve aliased inputs until the real launch."""

import os
import subprocess
import sys


def test_cold_mul_alias(tmp_path):
    code = r"""
import torch
import flag_gems
from flag_gems.ops.mul import mul, mul_
for shape in [(17, 16, 128), (19, 129), (23, 16, 128)]:
    host = torch.full((shape[0] * 2, *shape[1:]), .125, dtype=torch.bfloat16)
    for warm in range(2):
        x = host.to(flag_gems.device)
        view = x[::2]
        if shape[0] == 23:
            mul(view, torch.full_like(view, 16), out=view)
        elif len(shape) == 2:
            mul_(view, torch.full((1, shape[1]), 16., device=flag_gems.device, dtype=view.dtype))
        else:
            mul_(view, 16)
        actual = x.cpu()
        assert torch.equal(actual[::2], torch.full_like(host[::2], 2)), (shape, warm)
        assert torch.equal(actual[1::2], host[1::2]), (shape, warm, 'holes')
for warm in range(2):
    x = torch.full((10, 7), .125 + 0j, dtype=torch.complex64, device=flag_gems.device)
    mul_(x[::2], 16)
    actual = x.cpu()
    assert torch.equal(actual[::2], torch.full_like(actual[::2], 2))
    assert torch.equal(actual[1::2], torch.full_like(actual[1::2], .125))
"""
    env = dict(os.environ, FLAGGEMS_CACHE_DIR=str(tmp_path / "gems"))
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=180)
