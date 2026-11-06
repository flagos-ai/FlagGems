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

"""CI-only recorder for #8489; original assertions and dispatch stay unchanged."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SEED = 20261106
MAX_CACHE_FILES = 128
MAX_CACHE_FILE_BYTES = 16 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(v) for v in value)
    return value


def _copy_compiled(root):
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    copied = []
    for path in sorted(cache.rglob("_weight_gradient_packed.*")):
        if len(copied) >= MAX_CACHE_FILES:
            LOGGER.warning(
                "Compiler artifact capture reached %s files", MAX_CACHE_FILES
            )
            break
        if path.is_file() and path.stat().st_size <= MAX_CACHE_FILE_BYTES:
            dest = root / "compiled" / path.parent.name / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            copied.append(
                {"path": str(dest.relative_to(root)), "sha256": _sha256(dest)}
            )
    _write_json(root / "compiled-manifest.json", copied)


def _snapshot(root):
    import torch
    import torch_npu  # noqa: F401
    import triton

    import flag_gems

    source = Path(flag_gems.__file__).resolve().parent
    expected_source = Path.cwd() / "src/flag_gems"
    source_files = (
        "ops/convolution_backward.py",
        "runtime/backend/_ascend/ops/convolution_backward.py",
        "utils/libentry.py",
    )
    compiler = Path(triton.__file__).resolve().parent
    files = [
        source / "ops/convolution_backward.py",
        source / "runtime/backend/_ascend/ops/convolution_backward.py",
        source / "utils/libentry.py",
        compiler / "backends/ascend/compiler.py",
        compiler / "language/core.py",
        compiler / "_C/libtriton.so",
    ]
    tool = shutil.which("bishengir-compile")
    if tool:
        files.append(Path(tool).resolve())
    device = torch.npu.get_device_name(0)
    data = {
        "device": device,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-npu", "flagtree", "pytest")
        },
        "python": sys.version,
        "operator_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "gems_path": str(source),
        "triton_path": str(compiler),
        "target": repr(triton.runtime.driver.active.get_current_target()),
        "fp32_default": triton.knobs.language.fp32_default,
        "torch_npu_allow_hf32": torch.npu.matmul.allow_hf32,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "TRITON_F32_DEFAULT",
                "TRITON_ASCEND_ARCH",
                "TRITON_CACHE_DIR",
                "FLAGGEMS_CACHE_DIR",
                "ASCEND_RT_VISIBLE_DEVICES",
                "NPU_VISIBLE_DEVICES",
                "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME",
            )
        },
        "sha256": {str(path): _sha256(path) for path in files if path.is_file()},
    }
    _write_json(root / "runtime.json", data)
    result = subprocess.run(
        ["npu-smi", "info"], capture_output=True, text=True, timeout=30
    )
    (root / "npu-smi.txt").write_text(result.stdout + result.stderr)
    sdk = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
    sdk_files = [sdk / "version.cfg", sdk / "aarch64-linux/ascend_toolkit_install.info"]
    sdk_files.extend(sdk.glob("*/data/platform_config/Ascend910B[13].ini"))
    sdk_files.extend(sdk.glob("compiler/data/platform_config/Ascend910B[13].ini"))
    for path in sdk_files:
        if path.is_file():
            dest = root / "sdk" / path.relative_to(sdk)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    LOGGER.info("Runtime: %s", json.dumps(data))
    for relative in source_files:
        if _sha256(source / relative) != _sha256(expected_source / relative):
            raise RuntimeError(
                f"Installed source differs from pinned checkout: {relative}"
            )
    if data["operator_sha"] != os.environ["OPERATOR_SHA"]:
        raise RuntimeError(
            "Operator source revision differs from the failing CI revision"
        )
    if data["packages"]["flagtree"] != "0.7.1a1+ascend3.5":
        raise RuntimeError("Compiler version differs from the failing CI runtime")
    if "910B3" not in device:
        raise RuntimeError(f"Expected 910B3 for reproduction; got {device}")


def pytest_sessionstart(session):
    import torch

    # Restart the same input sequence in the cold and warm subprocesses.
    torch.manual_seed(SEED)
    torch.npu.manual_seed_all(SEED)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed or call.excinfo is None:
        return
    if "test_convolution_backward" not in item.nodeid:
        return
    import torch

    case = Path(os.environ["CONVBWD_DIAG_DIR"]) / os.environ["CONVBWD_PHASE"]
    case /= hashlib.sha256(item.nodeid.encode()).hexdigest()[:12]
    case.mkdir(parents=True, exist_ok=True)
    try:
        tb = call.excinfo.value.__traceback__
        saved = False
        while tb is not None:
            scope = tb.tb_frame.f_locals
            args = scope.get("args")
            if isinstance(args, (tuple, list)) and len(args) >= 10:
                if all(isinstance(t, torch.Tensor) for t in args[:3]):
                    torch.save(
                        {
                            key: _cpu(scope.get(key))
                            for key in ("args", "mask", "expected", "result")
                        },
                        case / "failure-inputs.pt",
                    )
                    saved = True
                    break
            tb = tb.tb_next
        _write_json(
            case / "failure.json", {"nodeid": item.nodeid, "inputs_captured": saved}
        )
    except Exception:
        # Recording must never replace the original test failure or its exit code.
        LOGGER.exception("Could not record failed inputs for %s", item.nodeid)


def _compare(root):
    sys.path.insert(0, str(Path.cwd()))

    import torch
    import torch_npu  # noqa: F401

    import flag_gems
    from tests.accuracy_utils import gems_assert_close
    from tests.test_convolution_backward import _inputs, _reference

    spec = importlib.util.spec_from_file_location(
        "upstream_convolution_backward", root / "upstream_convolution_backward.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    rows = []
    mask = (False, True, False)
    cases = list(range(8)) + sorted(root.glob("*/*/failure-inputs.pt"))
    for case in cases:
        if isinstance(case, int):
            torch.manual_seed(case)
            torch.npu.manual_seed_all(case)
            args = _inputs(3, False, 1, torch.float32)
            case_id = f"seed-{case}"
        else:
            payload = torch.load(case, weights_only=True)
            if tuple(payload["mask"] or ()) != mask:
                continue
            args = payload["args"]
            args = (*(tensor.to(flag_gems.device) for tensor in args[:3]), *args[3:])
            case_id = "-".join(case.relative_to(root).parts[:-1])
        expected = _reference(args)[1]
        saved = {
            "case": case_id,
            "args": _cpu(args),
            "mask": mask,
            "expected": expected,
        }
        for label, function in (
            ("pr", flag_gems.convolution_backward),
            ("upstream", upstream.convolution_backward),
        ):
            row = {"case": case_id, "implementation": label, "pass": False}
            try:
                actual = function(*args, mask)[1].cpu()
                saved[label] = actual
                row["max_abs"] = (actual.double() - expected).abs().max().item()
                gems_assert_close(actual, expected, torch.float32)
                row["pass"] = True
            except Exception as error:
                row["error"] = repr(error)
                LOGGER.exception(
                    "Comparison failed: case=%s implementation=%s", case_id, label
                )
            rows.append(row)
            LOGGER.info("Comparison: %s", json.dumps(row))
        torch.save(saved, root / f"compare/{case_id}.pt")
        _write_json(root / "compare/rows.json", rows)
    return int(any(not row["pass"] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("snapshot", "cold", "warm", "compare", "collect")
    )
    phase = parser.parse_args().phase
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    root = Path(os.environ["CONVBWD_DIAG_DIR"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ["CONVBWD_CACHE_ROOT"])
    if phase == "collect":
        for recorded_phase in ("cold", "warm", "compare"):
            cache_group = "compare" if recorded_phase == "compare" else "suite"
            os.environ["TRITON_CACHE_DIR"] = str(cache / cache_group / "triton")
            _copy_compiled(root / recorded_phase)
        return 0
    cache_group = "suite" if phase in ("cold", "warm") else phase
    os.environ["TRITON_CACHE_DIR"] = str(cache / cache_group / "triton")
    os.environ["FLAGGEMS_CACHE_DIR"] = str(cache / cache_group / "gems")
    if phase == "snapshot":
        _snapshot(root)
        return 0
    (root / phase).mkdir(parents=True, exist_ok=True)
    os.environ["CONVBWD_PHASE"] = phase
    status = 1
    try:
        if phase == "compare":
            status = _compare(root)
        else:
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                (
                    str(Path(__file__).resolve().parent),
                    str(Path.cwd()),
                    env.get("PYTHONPATH", ""),
                )
            )
            status = subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "-m",
                    "pytest",
                    "-p",
                    Path(__file__).stem,
                    "-s",
                    "-v",
                    "tests/test_convolution_backward.py",
                    "--tb=short",
                    "-x",
                    f"--junitxml={root / phase / 'junit.xml'}",
                ],
                env=env,
            )
        return status
    finally:
        _write_json(
            root / phase / "status.json",
            {
                "exit_code": status,
                "seeds": list(range(8)) if phase == "compare" else [SEED],
            },
        )
        _copy_compiled(root / phase)


if __name__ == "__main__":
    sys.exit(main())
