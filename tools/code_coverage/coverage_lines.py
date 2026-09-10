#!/usr/bin/env python


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

import os
import sys


def get_lines(info_file):
    hits, total = 0.0, 0.0
    with open(info_file) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("DA:"):
                continue
            total += 1
            if int(line[3:].split(",")[1]) > 0:
                hits += 1
    if total == 0:
        print("no data found")
        sys.exit()
    return hits / total


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: coverage_lines.py info_file expected")
        sys.exit(1)

    info_file, expected = sys.argv[1], float(sys.argv[2])

    if not os.path.isfile(info_file):
        print(f"info file {info_file} is not exists, ignored")
        sys.exit(1)
    actual = round(get_lines(info_file), 3)

    if actual < expected:
        print(
            f"expected >= {round(expected * 100, 1)} %, actual {round(actual * 100, 1)} %, failed"
        )
        print("\n================================================================")
        sys.exit(1)

    print(
        f"expected >= {round(expected * 100, 1)} %, actual {round(actual * 100, 1)} %, passed"
    )
    print("\n================================================================")
