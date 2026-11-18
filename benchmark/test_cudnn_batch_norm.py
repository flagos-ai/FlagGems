# Copyright 2026, The FlagOS Contributors.
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
#
import json
import signal
import subprocess
import sys

import pytest
import torch

import flag_gems

from . import base
from .conftest import Config, recordLogger

# Run only pure PyTorch in the child; importing this benchmark would load FlagGems.
_NATIVE_OUT_PROBE = """
import json
import logging
import sys

logger = logging.getLogger(__name__)

def _run_probe(config):
    import gc

    import torch

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logger.info("torch=%s cuda=%s", torch.__version__, torch.version.cuda)
    device = config["device"]
    for dtype_name in config["dtypes"]:
        dtype = getattr(torch, dtype_name)
        param_dtype = torch.float32 if dtype == torch.float16 else dtype
        for shape in config["shapes"]:
            for training in (True, False):
                x = torch.randn(shape, dtype=dtype, device=device)
                w = torch.ones(shape[1], dtype=param_dtype, device=device)
                b = torch.zeros_like(w)
                rm, rv = b.clone(), w.clone()
                outputs = dict(
                    out0=torch.empty_like(x),
                    out1=torch.empty(
                        shape[1] if training else 0, dtype=param_dtype, device=device
                    ),
                    out2=torch.empty(
                        shape[1] if training else 0, dtype=param_dtype, device=device
                    ),
                    out3=torch.empty(0, dtype=torch.uint8, device=device),
                )
                logger.info("%s %s %s %s", config["started_marker"], dtype_name, shape, training)
                for _ in range(config["repetitions"]):
                    # Match benchmark timing: discard each returned tuple.
                    torch.ops.aten.cudnn_batch_norm.out(
                        x, w, b, rm, rv, training, 0.1, 1e-5, **outputs
                    )
                torch.cuda.synchronize(device)
                del outputs, x, w, b, rm, rv
                gc.collect()
    assert "flag_gems" not in sys.modules
    logger.info(config["completed_marker"])
    # Normal interpreter shutdown is part of the probe; never use os._exit().


_run_probe(json.load(sys.stdin))
"""

_PROBE_REPETITIONS = 32

_PROBE_STARTED = "NATIVE_CUDNN_BATCH_NORM_OUT_STARTED"
_PROBE_COMPLETED = "NATIVE_CUDNN_BATCH_NORM_OUT_COMPLETED"


def _probe_native_out(shapes, dtypes, device):
    return subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", _NATIVE_OUT_PROBE],
        input=json.dumps(
            {
                "shapes": shapes,
                "repetitions": _PROBE_REPETITIONS,
                "started_marker": _PROBE_STARTED,
                "completed_marker": _PROBE_COMPLETED,
                "dtypes": [str(dtype).removeprefix("torch.") for dtype in dtypes],
                "device": str(device),
            }
        ),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _native_out_skip_reason(result):
    if result.returncode == 0 and _PROBE_COMPLETED in result.stdout:
        return None
    # Do not classify an import/device setup failure as the known native bug.
    if _PROBE_STARTED in result.stdout:
        if result.returncode in (-signal.SIGSEGV, -signal.SIGABRT):
            return (
                "native cudnn_batch_norm.out lifetime probe terminated with "
                f"{signal.Signals(-result.returncode).name}; no valid native baseline"
            )
        if result.returncode != 0 and "pyobject_preservation.cpp" in result.stderr:
            return (
                "native cudnn_batch_norm.out failed PyTorch object preservation; "
                "no valid native baseline"
            )
    raise RuntimeError(
        f"native cudnn_batch_norm.out probe failed ({result.returncode})\n"
        f"{result.stdout}\n{result.stderr}"
    )


pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name in ("mthreads", "ascend", "hygon"),
    reason=(
        "native cudnn_batch_norm has no MUSA kernel, falls back to CPU on Ascend, "
        "and is not compiled with cuDNN support on Hygon"
    ),
)

DTYPES = [torch.float16, torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)


class CudnnBatchNormBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [
        (16, 32),
        (16, 16, 64),
        (16, 16, 1024),
        (16, 16, 4098),
        (1, 8, 4, 4),
        (16, 8, 128, 128),
        (2, 16, 7, 9, 11),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(self.DEFAULT_SHAPES)
        self.shape_desc = "N, C, spatial dimensions"


def _inputs(shape, dtype, device, out=False):
    param_dtype = torch.float32 if dtype == torch.float16 else dtype
    for training in (True, False):
        x = torch.randn(shape, dtype=dtype, device=device)
        w = torch.randn(shape[1], dtype=param_dtype, device=device)
        b = torch.randn_like(w)
        rm = torch.zeros_like(w)
        rv = torch.ones_like(w)
        args = (x, w, b, rm, rv, training, 0.1, 1e-5)
        if out:
            kwargs = dict(
                out0=torch.empty_like(x),
                out1=torch.empty(
                    shape[1] if training else 0, dtype=param_dtype, device=device
                ),
                out2=torch.empty(
                    shape[1] if training else 0, dtype=param_dtype, device=device
                ),
                out3=torch.empty(0, dtype=torch.uint8, device=device),
            )
            yield (*args, kwargs)
        else:
            yield args


def _out_inputs(shape, dtype, device):
    yield from _inputs(shape, dtype, device, out=True)


@pytest.mark.cudnn_batch_norm
def test_cudnn_batch_norm():
    bench = CudnnBatchNormBenchmark(
        input_fn=_inputs,
        op_name="cudnn_batch_norm",
        torch_op=torch.ops.aten.cudnn_batch_norm.default,
        dtypes=DTYPES,
    )
    bench.set_gems(flag_gems.cudnn_batch_norm)
    bench.run()


@pytest.mark.cudnn_batch_norm_out
def test_cudnn_batch_norm_out():
    if (
        flag_gems.vendor_name == "nvidia"
        and not Config.skip_native
        and not (Config.query or Config.list_cases or Config.profile_only)
    ):
        result = _probe_native_out(
            CudnnBatchNormBenchmark.DEFAULT_SHAPES,
            Config.user_desired_dtypes or DTYPES,
            torch.device("cuda", torch.cuda.current_device()),
        )
        if Config.record_log and result.returncode:
            # Keep diagnostics outside the INFO JSON latency records.
            recordLogger.warning(
                "native_cudnn_batch_norm_out_probe=%s",
                json.dumps(
                    {
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                ),
            )
        reason = _native_out_skip_reason(result)
        if reason is not None:
            # The autouse configure_native_baseline fixture restores these per test.
            Config.skip_native = True
            Config.native_baseline_skip_reason = reason
    bench = CudnnBatchNormBenchmark(
        input_fn=_out_inputs,
        op_name="cudnn_batch_norm_out",
        torch_op=torch.ops.aten.cudnn_batch_norm.out,
        dtypes=DTYPES,
    )
    bench.set_gems(flag_gems.cudnn_batch_norm_out)
    bench.run()
