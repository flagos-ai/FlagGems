# Copyright 2026 FlagOS Contributors
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

"""Run real pytest.raises checks; isolate each poisoned device context."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

CHILD = "IX_PYTEST_ERROR_CHILD"


def _isolated(request):
    if os.environ.get(CHILD) == "1":
        return False
    node = str(Path(__file__).resolve()) + "::" + request.node.name
    env = dict(os.environ, **{CHILD: "1", "TRITON_DEBUG": "0"})
    command = [sys.executable, "-m", "pytest", "-c", "/dev/null", "-s", "-vv", node]
    with subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            pytest.fail("Child pytest timed out:\n" + output)
    print(output, flush=True)
    assert process.returncode == 0, f"Child pytest exit={process.returncode}\n{output}"
    assert "1 passed" in output, output
    return True


def _backward_inputs():
    import torch

    device = "cuda"
    grad = torch.ones((2, 33), device=device)
    indices = torch.tensor([1, 2, 1], device=device)
    offsets = torch.tensor([0, 2], device=device)
    mapping = torch.tensor([0, 0, 1], device=device)
    sizes = torch.tensor([2, 1], device=device)
    maximum = torch.empty(0, device=device, dtype=torch.int64)
    return grad, indices, offsets, mapping, sizes, maximum, 8, False, 1, False


def _report(caught):
    error = caught.value
    print(f"CAUGHT {type(error).__module__}.{type(error).__name__}", flush=True)
    print(f"MRO {type(error).__mro__}", flush=True)
    print(str(error), flush=True)


@pytest.mark.parametrize("error_name", ["RuntimeError", "AcceleratorError"])
def test_python_exception_type(request, error_name):
    if _isolated(request):
        return
    import torch

    error_type = (
        RuntimeError if error_name == "RuntimeError" else torch.AcceleratorError
    )
    assert issubclass(error_type, RuntimeError)
    with pytest.raises(RuntimeError, match="pytest type control") as caught:
        raise error_type("pytest type control")
    _report(caught)


@pytest.mark.parametrize("after_backward", [False, True])
@pytest.mark.parametrize("keep_flag", [False, True])
@pytest.mark.parametrize("implementation", ["native", "helper_scalar", "helper_vector"])
def test_device_assert(request, implementation, keep_flag, after_backward):
    if _isolated(request):
        return
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def native(flag):
        code = tl.load(flag)
        tl.device_assert(code == 0, "IX pytest device assert")

    if implementation == "native":

        def check(flag):
            native[(1,)](flag, debug=True)

    else:
        from flag_gems.runtime.backend._iluvatar.ops._embedding_bag import (
            _check_native_error,
        )

        def check(flag):
            _check_native_error(flag, flag.numel())

    count = 513 if implementation == "helper_vector" else 1
    flag = torch.zeros(count, dtype=torch.int32, device="cuda")
    # Compile and exercise the same specialization on valid input outside raises.
    check(flag)
    torch.cuda.synchronize()
    if after_backward:
        from flag_gems.runtime.backend._iluvatar.ops._embedding_bag_backward import (
            _embedding_bag_backward,
        )

        result = _embedding_bag_backward(*_backward_inputs())
        torch.cuda.synchronize()
        assert result.shape == (8, 33)
    flag[-1] = 1
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match="(?i)assert") as caught:
        check(flag)
        if not keep_flag:
            del flag
        torch.cuda.synchronize()
    _report(caught)


@pytest.mark.parametrize("from_forward", [False, True])
def test_backward_invalid_index(request, from_forward):
    if _isolated(request):
        return
    import torch

    from flag_gems.runtime.backend._iluvatar.ops._embedding_bag import _embedding_bag
    from flag_gems.runtime.backend._iluvatar.ops._embedding_bag_backward import (
        _embedding_bag_backward,
    )

    args = list(_backward_inputs())
    if from_forward:
        weight = torch.ones((8, 33), device="cuda")
        output, mapping, sizes, maximum = _embedding_bag(
            weight, args[1], args[2], mode=1
        )
        args[:1] = [torch.ones_like(output)]
        args[3:6] = [mapping, sizes, maximum]
    result = _embedding_bag_backward(*args)
    torch.cuda.synchronize()
    assert result.shape == (8, 33)
    args[1][0] = 8
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match="(?i)assert") as caught:
        result = _embedding_bag_backward(*args)
        torch.cuda.synchronize()
    _report(caught)
