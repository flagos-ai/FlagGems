# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import concurrent.futures
import multiprocessing
import time
from pathlib import Path

from flag_gems.utils.code_utils import file_lock


def _append_with_lock(args):
    path, value = args
    path = Path(path)
    with file_lock(path):
        lines = path.read_text(encoding="utf-8").splitlines()
        time.sleep(0.01)
        path.write_text("\n".join([*lines, value]) + "\n", encoding="utf-8")


def test_file_lock_serializes_shared_cache_updates(tmp_path):
    path = tmp_path / "generated.py"
    path.write_text("", encoding="utf-8")
    values = [f"value-{index}" for index in range(16)]
    context = multiprocessing.get_context("spawn")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=8, mp_context=context
    ) as executor:
        list(executor.map(_append_with_lock, [(str(path), value) for value in values]))

    assert sorted(path.read_text(encoding="utf-8").splitlines()) == sorted(values)
