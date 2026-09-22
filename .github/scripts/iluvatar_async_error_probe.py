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

"""Diagnostic only: every device fault runs in a fresh, bounded subprocess."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

METHODS = ("native", "sdk", "trap")
PREFIX = "IX_PROBE "


def emit(**record):
    print(PREFIX + json.dumps(record), flush=True)


def child(method, invalid, store_after):
    import torch
    import triton
    import triton.language as tl
    import triton.language.core as tlcore
    from triton.runtime.cache import get_cache_manager

    from flag_gems.runtime.backend._iluvatar.ops._embedding_bag import (
        _corex_assert_library,
        _embedding_bag_corex_assert,
    )

    # This is a compiler experiment, not an assumed-supported IX instruction.
    # A compilation failure is reported separately from a runtime exception.
    trap_ir = """target triple = "bi-iluvatar-ilurt"
    declare void @llvm.trap()
    define i32 @embedding_bag_probe_trap(i32 %code) {
    entry:
      %ok = icmp eq i32 %code, 0
      br i1 %ok, label %end, label %invalid
    invalid:
      call void @llvm.trap()
      unreachable
    end:
      ret i32 %code
    }
    """

    @triton.jit
    def probe(flag, output, kind: tl.constexpr, write_after: tl.constexpr):
        code = tl.load(flag)
        if kind == 0:
            tl.device_assert(code == 0, "IX asynchronous error probe")
        elif kind == 1:
            _embedding_bag_corex_assert(code)
        else:
            tlcore.extern_elementwise(
                "",
                "",
                [code],
                {(tl.int32,): ("embedding_bag_probe_trap", tl.int32)},
                is_pure=False,
            )
        if write_after:
            tl.store(output, 11)

    options = {"debug": True}
    if method == "sdk":
        options["extern_libs"] = {"assert_probe": _corex_assert_library()}
    elif method == "trap":
        key = hashlib.sha256((triton.__version__ + trap_ir).encode()).hexdigest()
        cache = get_cache_manager(key)
        path = cache.get_file("probe.ll")
        if path is None:
            path = cache.put(trap_ir, "probe.ll", binary=False)
        options["extern_libs"] = {"trap_probe": path}
    stage = "warmup"
    try:
        flag = torch.zeros((), device="cuda", dtype=torch.int32)
        output = torch.zeros((), device="cuda", dtype=torch.int32)
        # Compile the exact same specialization on a valid input first.
        probe[(1,)](flag, output, METHODS.index(method), store_after, **options)
        torch.cuda.synchronize()
        if store_after and output.item() != 11:
            raise RuntimeError("Valid warmup did not produce the expected output")
        flag.fill_(int(invalid))
        torch.cuda.synchronize()
        stage = "launch"
        emit(stage=stage, torch=torch.__version__, triton=triton.__version__)
        probe[(1,)](flag, output, METHODS.index(method), store_after, **options)
        stage = "synchronize"
        emit(stage=stage)
        torch.cuda.synchronize()
    except Exception as exc:
        status = (
            "runtime_error"
            if stage != "warmup" and isinstance(exc, RuntimeError)
            else "setup_or_compile_error"
        )
        emit(status=status, stage=stage, error=repr(exc))
        return
    emit(status="missing_error" if invalid else "valid_ok", stage=stage)


def run(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for method in METHODS:
        for store_after in (False, True):
            for invalid in (False, True):
                name = f"{method}-after{int(store_after)}-invalid{int(invalid)}"
                command = [sys.executable, __file__, "--child", method]
                if invalid:
                    command.append("--invalid")
                if store_after:
                    command.append("--store-after")
                with (output_dir / f"{name}.log").open("w") as log:
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        returncode = process.wait(timeout=90)
                        status = "process_failure" if returncode else "missing_record"
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        returncode, status = process.returncode, "timeout"
                records = []
                for line in (output_dir / f"{name}.log").read_text().splitlines():
                    if line.startswith(PREFIX):
                        records.append(json.loads(line[len(PREFIX) :]))
                if returncode == 0:
                    status = next(
                        (r["status"] for r in reversed(records) if "status" in r),
                        status,
                    )
                result = dict(
                    method=method,
                    store_after=store_after,
                    invalid=invalid,
                    status=status,
                    returncode=returncode,
                    records=records,
                )
                results.append(result)
                print(json.dumps(result), flush=True)
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    # Do not accept a printed assertion or process abort as a Torch exception.
    reliable = [
        method
        for method in METHODS
        if all(
            r["status"] == ("runtime_error" if r["invalid"] else "valid_ok")
            for r in results
            if r["method"] == method
        )
    ]
    print("Reliable candidates:", reliable, flush=True)
    return 0 if reliable else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=METHODS)
    parser.add_argument("--invalid", action="store_true")
    parser.add_argument("--store-after", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("ix-probe-results"))
    args = parser.parse_args()
    if args.child:
        child(args.child, args.invalid, args.store_after)
    else:
        sys.exit(run(args.output))
