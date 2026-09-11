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

import argparse
import subprocess

import yaml


def load_shape_file(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_markers(shape_data):
    return list({op_name for op_name, shapes in shape_data.items() if shapes})


def run_benchmark_command(markers_str, shape_file, args):
    cmd = [
        "pytest",
        "-m",
        markers_str,
        "-s",
        "--level",
        "core",
        "--record",
        "log",
        "--shape_file",
        shape_file,
    ]
    if args.extra_args:
        cmd += args.extra_args.split()

    result = subprocess.run(cmd, capture_output=True, text=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark for given operators list and corresponding shapes."
    )
    parser.add_argument(
        "--shape-file",
        type=str,
        default="shapes.yaml",
        help="Path to the shape file (default: shapes.yaml)",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Extra args to pass to pytest (e.g., '--tb=short')",
    )
    args = parser.parse_args()

    shape_data = load_shape_file(args.shape_file)
    markers = build_markers(shape_data)
    if not markers:
        print(f"[Warning] No markers found in {args.shape_file}, skip benchmarking.")
        exit(0)

    markers_str = " or ".join(markers)
    run_benchmark_command(markers_str, args.shape_file, args)
