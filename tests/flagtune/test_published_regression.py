"""Check real model selection, kernel correctness and AUTO fallback.

Run on an idle device with explicit opt-in; select operators with pytest -k::

    FLAGTUNE_RUN_PUBLISHED_REGRESSION=1 CUDA_VISIBLE_DEVICES=0 \\
      python -m pytest tests/flagtune/test_published_regression.py -v -k mul

Each case runs in a fresh subprocess with private model, Triton and FlagGems
caches. The published scenario requires real Cost Model proposals; the other
scenarios require AUTO fallback. Both paths verify numerical correctness.
"""

import argparse
import importlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("FLAGTUNE_RUN_PUBLISHED_REGRESSION") != "1",
    reason="set FLAGTUNE_RUN_PUBLISHED_REGRESSION=1 on an idle device",
)
@pytest.mark.parametrize(
    "scenario", ["published", "unavailable", "incompatible", "missing-variant"]
)
@pytest.mark.parametrize("operator", ["mul", "mm", "mv"])
def test_published_regression(tmp_path, scenario, operator):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "report.json"
    environment = os.environ.copy()
    environment.update(
        FLAGTUNE_MODEL_CACHE=str(tmp_path / "models"),
        TRITON_CACHE_DIR=str(tmp_path / "triton"),
        FLAGGEMS_CACHE_DIR=str(tmp_path / "flaggems"),
        PYTHONPATH=os.pathsep.join(
            path for path in (str(root / "src"), environment.get("PYTHONPATH")) if path
        ),
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--operators",
            operator,
            "--scenario",
            scenario,
            "--output",
            str(output),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    report = json.loads(output.read_text())
    assert report["scenario"] == scenario
    assert report["checks"]
    assert all(check["correct"] for check in report["checks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operators", nargs="+", choices=["mul", "mm", "mv"], default=["mul"]
    )
    parser.add_argument(
        "--scenario",
        choices=["published", "unavailable", "incompatible", "missing-variant"],
        default="published",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.pop("USE_FLAGTUNE", None)
    os.environ["FLAGTUNE_INCLUDE"] = ",".join(args.operators)
    os.environ.setdefault("FLAGTUNE_TOP_K", "2")
    if args.scenario != "published":
        manifest = args.output.with_suffix(".manifest.json")
        manifest.write_text(json.dumps({"schema_version": 1, "packages": {}}))
        os.environ["FLAGTUNE_LOCAL_MANIFEST"] = str(manifest.resolve())
        os.environ["FLAGTUNE_MODEL_VERSION"] = "999.0.0"
        os.environ["FLAGTUNE_DISABLE_REMOTE"] = "1"
        os.environ.pop("FLAGTUNE_MODEL_DIR", None)
        os.environ.pop("USE_FLAGTUNE_COST_MODEL", None)
        os.environ.pop("FLAGTUNE_INCLUDE", None)
        if args.scenario != "unavailable":
            # Probe the actual package reader and missing-child path. Neither
            # scenario replaces model validation or the AUTO policy.
            model_root = args.output.with_suffix(".models")
            model_root.mkdir(parents=True, exist_ok=True)
            os.environ["FLAGTUNE_MODEL_DIR"] = str(model_root.resolve())
            from triton.flagtune.contract.identity import discover_gpu_metadata

            platform = discover_gpu_metadata()["platform_key"]
            package = model_root / f"{platform}_v999.0.0.tar.gz"
            if args.scenario == "incompatible":
                package.write_bytes(b"invalid model archive")
            else:
                payload = json.dumps(
                    {
                        "schema_version": 1,
                        "platform_key": platform,
                        "package_version": "999.0.0",
                        "models": {},
                    }
                ).encode()
                with tarfile.open(package, "w:gz") as archive:
                    member = tarfile.TarInfo("package_manifest.json")
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
    else:
        os.environ["USE_FLAGTUNE_COST_MODEL"] = "1"

    import torch
    import triton
    from triton.flagtune import _version

    import flag_gems
    from flag_gems.flagtune.inference import cost_model

    assert tuple(map(int, _version.__version__.split("."))) >= (
        1,
        1,
        0,
    ), _version.__version__
    libentry = importlib.import_module("flag_gems.utils.libentry")
    events = []
    original_propose = cost_model.run_proposer
    original_fallback = libentry._flagtune_legacy_fallback

    def record_proposal(tuner, benchmark, candidates, arguments, loaded):
        selected, timings = original_propose(
            tuner, benchmark, candidates, arguments, loaded
        )
        identity, _, variant = loaded
        (config,) = cost_model.configs_to_dicts([selected], list(variant.param_names))
        assert variant.param_space.validate(config), (identity, config)
        events.append(
            {
                "event": "prediction",
                "identity": identity.artifact_key,
                "config": config,
                "runtime_candidates": len(candidates),
            }
        )
        return selected, timings

    def record_fallback(tuner, benchmark, positional, keywords, op_name):
        result = original_fallback(tuner, benchmark, positional, keywords, op_name)
        events.append({"event": "fallback", "op": op_name})
        return result

    cost_model.run_proposer = record_proposal
    libentry._flagtune_legacy_fallback = record_fallback
    torch.manual_seed(0)
    checks = []
    device = flag_gems.device
    cases = []
    if "mul" in args.operators:
        x = torch.randn((17, 129), device=device, dtype=torch.bfloat16)
        rhs = torch.randn((129,), device=device, dtype=torch.bfloat16)
        cases.extend(
            [("mul_broadcast", "mul", (x, rhs)), ("mul_scalar", "mul", (x, 0.5))]
        )
    if "mm" in args.operators:
        for m, n, k in [(128, 128, 256), (32, 64, 128)]:
            a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
            b = torch.randn((k, n), device=device, dtype=torch.bfloat16)
            cases.append((f"mm_{m}_{n}_{k}", "mm", (a, b)))
    if "mv" in args.operators:
        a = torch.randn((32, 256), device=device, dtype=torch.bfloat16)
        x = torch.randn((256,), device=device, dtype=torch.bfloat16)
        cases.append(("mv", "mv", (a, x)))
    for name, operator, operands in cases:
        reference = getattr(torch, operator)(*operands)
        before = len(events)
        with flag_gems.use_gems(include=[operator]):
            actual = getattr(torch, operator)(*operands)
        torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.05)
        observed = events[before:]
        expected = "prediction" if args.scenario == "published" else "fallback"
        assert any(event["event"] == expected for event in observed), (name, observed)
        checks.append({"case": name, "correct": True, "events": observed})

    report = {
        "scenario": args.scenario,
        "flagtune_version": _version.__version__,
        "flagtune_source": _version.__file__,
        "triton_source": triton.__file__,
        "flaggems_source": flag_gems.__file__,
        "device": str(device),
        "torch_version": torch.__version__,
        "checks": checks,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
