#!/bin/bash


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

# Configuration parameters
mem_threshold=30000     # Minimum free memory required (MB)
sleep_time=120          # Wait time between retries (seconds)
max_wait=600            # Maximum total wait time (seconds)

# Check if rocm-smi exists
if ! command -v rocm-smi &> /dev/null; then
    echo "Error: rocm-smi command not found. Please check if ROCm is installed."
    exit 1
fi

waited_time=0
while true; do
    # rocm-smi reports VRAM in bytes via JSON; keys differ across ROCm
    # versions, so match on lowercased substrings ("total memory" /
    # "used memory") and convert to MiB. The per-GPU table is printed to
    # stderr for debugging; only the "Available GPUs:" line goes to stdout,
    # which the weekly workflow greps for.
    AVAILABLE_GPUS=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c '
import json, sys

mem_threshold = 30000  # MiB
try:
    data = json.load(sys.stdin)
except Exception:
    print("Failed to parse rocm-smi JSON output.", file=sys.stderr)
    sys.exit(2)

avail = []
print(" GPU  Total (MiB)  Used (MiB)  Free (MiB)", file=sys.stderr)
for i, (card, info) in enumerate(sorted(data.items())):
    total_b = used_b = None
    for k, v in info.items():
        kl = k.lower()
        if "total memory" in kl:
            total_b = int(v)
        elif "used memory" in kl:
            used_b = int(v)
    if total_b is None or used_b is None:
        continue
    total = total_b // (1024 * 1024)
    used = used_b // (1024 * 1024)
    free = total - used
    print("%4d %12d %11d %11d" % (i, total, used, free), file=sys.stderr)
    if free >= mem_threshold:
        avail.append(str(i))

print(",".join(avail))
')
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "Failed to query GPU memory information via rocm-smi."
        exit 1
    fi

    if [ -n "${AVAILABLE_GPUS}" ]; then
        echo "Available GPUs: ${AVAILABLE_GPUS}"
        break
    fi

    echo "No GPU has sufficient memory, waiting for $sleep_time seconds..."
    sleep $sleep_time
    waited_time=$((waited_time + sleep_time))
    if [ $waited_time -ge $max_wait ]; then
        echo "Error: Timed out waiting for available GPU."
        exit 1
    fi
done
