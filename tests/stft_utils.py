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

import json
import subprocess
import sys

import pytest
import torch

import flag_gems


@pytest.fixture(scope="module")
def ascend_sip_unavailable_reason():
    if flag_gems.vendor_name != "ascend":
        return None
    # SiP ships precompiled libraries. Probe their ABI and actual R2C/C2C
    # execution in a child process so a vendor crash cannot corrupt pytest.
    probe = r"""
import json
import sys
import torch
import flag_gems
from flag_gems.runtime.backend._ascend import stft_fft

def unavailable(error):
    print("SIP_UNAVAILABLE=" + json.dumps(str(error)))
    sys.exit(42)

try:
    stft_fft._LIBRARY = stft_fft._Library()
except (OSError, AttributeError) as error:
    unavailable(error)

device = "npu:" + sys.argv[1]
real = torch.zeros((2, 8), device=device, dtype=torch.float32)
complex_input = torch.zeros((2, 8), device=device, dtype=torch.complex64)
try:
    stft_fft.fft_frames(real)
    stft_fft.fft_frames(complex_input)
    stft_fft.fft_frames(complex_input, inverse=True)
    torch.npu.synchronize(device)
    for plan in stft_fft._PLANS.values():
        plan.close()
    stft_fft._PLANS.clear()
except RuntimeError as error:
    unavailable(error)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe, str(torch.npu.current_device())],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "SiP FFT execution probe timed out after 120 seconds"
    if result.returncode == 0:
        return None
    if result.returncode == 42:
        for line in result.stdout.splitlines():
            if line.startswith("SIP_UNAVAILABLE="):
                return "SiP FFT unavailable: " + json.loads(line.split("=", 1)[1])
    if result.returncode < 0:
        return f"SiP FFT probe terminated by signal {-result.returncode}"
    # Import errors and bugs in the probe are test failures, not missing-SDK
    # skips. Actual STFT assertions run normally after a successful probe.
    pytest.fail(
        f"Unexpected SiP probe failure (exit {result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )


@pytest.fixture
def require_stft_fft(request):
    def require(n_fft, *, backward=False, complex_half=False):
        if flag_gems.vendor_name != "ascend":
            return
        from flag_gems.runtime.backend._ascend.stft_fft import _fused_stft_supported

        # These tests use effective float32/complex64. Match the forward
        # eligibility guard; every nonempty backward still calls SiP FFT.
        fused = (
            not backward
            and not complex_half
            and 64 <= n_fft <= 1024
            and n_fft & (n_fft - 1) == 0
            and _fused_stft_supported()
        )
        if not fused:
            reason = request.getfixturevalue("ascend_sip_unavailable_reason")
            if reason is not None:
                pytest.skip(reason)

    return require
