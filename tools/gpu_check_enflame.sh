#!/bin/bash
set -x

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
sleep_interval=120      # Wait time between retries (seconds)
max_wait=600           # Maximum total wait time (seconds)

# Count chip lines (lines with Bus info)
gpu_count=$(efsmi -L | grep -cP '\d+\:\d+\:')

if [ "$gpu_count" -eq 0 ]; then
    echo "No Enflame GPUs detected. Please ensure you have Enflame GPUs installed and properly configured."
    exit 1
fi

echo "Detected $gpu_count Enflame GPU chip(s)."

# efsmi only talks to the driver's query interface: it never opens a device or
# creates a GCU context. A container can therefore report a card with plenty of
# free memory while torch still cannot alloc on it, because context creation
# (dtu_context_create -> ConfigDomain) needs device access the container does
# not have. Probe an actual allocation so that failure surfaces here, with a
# readable reason, instead of as an opaque driver abort inside the test run.
#
# The runtime image installs under /flagos; prefer that interpreter but fall
# back to whatever python3 the (login) shell resolves, so this stays correct if
# the image layout changes.
python_bin=/flagos/bin/python3
[ -x "$python_bin" ] || python_bin=$(command -v python3)

gcu_probe() {
    "$python_bin" - <<'PY'
import sys

import torch

gcu = getattr(torch, "gcu", None)
if gcu is None:
    print("torch has no 'gcu' module (torch-gcu not installed?)")
    sys.exit(1)

# Guarded: a missing is_available would otherwise read as "device unusable"
# rather than "this torch build exposes a different API".
if hasattr(gcu, "is_available") and not gcu.is_available():
    print("torch.gcu.is_available() is false")
    sys.exit(1)

# Device ordinals are assigned in discovery order, which need not match the
# efsmi indices, so probe the first exposed device rather than a fixed index.
dev = torch.device("gcu", 0)
try:
    torch.zeros(1, device=dev)
    # Make the failure surface here rather than at process teardown.
    sync = getattr(gcu, "synchronize", None)
    if sync is not None:
        sync()
except Exception as exc:
    print(f"allocating on {dev} failed: {type(exc).__name__}: {exc}")
    sys.exit(1)

print(f"allocation on {dev} succeeded")
PY
}

probe_out=$(gcu_probe 2>&1)
probe_rc=$?
if [ $probe_rc -ne 0 ]; then
    # A broken interpreter and an unusable device both land here. Distinguish
    # them so the message points at the actual cause instead of always blaming
    # the device.
    if ! "$python_bin" -c 'pass' >/dev/null 2>&1; then
        echo "Error: cannot run the GCU probe — ${python_bin} is not usable."
        echo "  ${probe_out}"
        exit 1
    fi
    echo "Error: Enflame GCU is visible to efsmi but unusable from this container."
    echo "  ${probe_out}"
    echo "  Context creation needs the GCU device nodes and driver domain config"
    echo "  inside the job's container, and this one does not have them."
    echo "  On the k8s runner this job uses, container.options is not honoured"
    echo "  (there is no docker create to pass it to), so --privileged / -v /dev"
    echo "  cannot fix this from the workflow — the pod spec used by the runner"
    echo "  has to expose the GCU devices."
    echo
    echo "--- Diagnostics: device nodes and container privilege ---"
    # Tells apart "the device nodes never reached this container" (nothing
    # listed / cgroup denies them) from "they're here but the driver still
    # can't init them" (nodes present, cgroup allows them, still fails) —
    # the two root causes need different fixes (pod spec vs. driver/domain).
    dtu_nodes=$(ls -la /dev 2>/dev/null | grep -iE 'dtu|gcu|efml|dri' || true)
    if [ -n "$dtu_nodes" ]; then
        echo "GCU-related device nodes under /dev:"
        echo "$dtu_nodes" | sed 's/^/  /'
    else
        echo "No GCU-related device nodes (dtu|gcu|efml|dri) found under /dev."
        echo "  -> the device was very likely never passed into this container."
    fi
    if [ -r /proc/self/status ]; then
        cap=$(grep -i '^CapEff:' /proc/self/status || true)
        echo "Capabilities: ${cap:-<unreadable>}"
    fi
    if [ -r /sys/fs/cgroup/devices/devices.list ]; then
        echo "Allowed devices (cgroup v1 devices.list):"
        sed 's/^/  /' /sys/fs/cgroup/devices/devices.list
    elif [ -r /sys/fs/cgroup/devices.list ]; then
        echo "Allowed devices (cgroup v2 devices.list):"
        sed 's/^/  /' /sys/fs/cgroup/devices.list
    else
        echo "Could not read a cgroup devices.list (cgroup v1 or v2) to check allowed devices."
    fi
    echo "--- End diagnostics ---"
    exit 1
fi

echo "GCU usable from this container ($probe_out)."

waited_time=0
while true; do
    available_gpus=()

    printf " GPU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    for ((i=0; i<$gpu_count; i++)); do
        total_i=$(efsmi -i $i -q -d MEMORY | awk '/Device Mem Info/,/BAR1/ { if (/Total Size/) {gsub(/[^0-9]/,"",$0); print $0}}')
        free_i=$(efsmi -i $i -q -d MEMORY | awk '/Device Mem Info/,/BAR1/ { if (/Free Size/) {gsub(/[^0-9]/,"",$0); print $0}}')

        if [ -z "$free_i" ] || [ -z "$total_i" ]; then
            echo "Warning: Failed to parse memory info for chip $i."
            continue
        fi

        used_i=$((total_i - free_i))

        printf "%4d%'13d%'12d%'12d\n" $i ${total_i} ${used_i} ${free_i}
        if [ $free_i -ge $mem_threshold ]; then
            available_gpus+=($i)
        fi
    done

    if [ ${#available_gpus[@]} -gt 0 ]; then
        AVAILABLE_GPUS=$(IFS=,; echo "${available_gpus[*]}")
        echo "Available GPUs: ${AVAILABLE_GPUS}"
        break
    fi

    echo "No GPU has sufficient memory, waiting for $sleep_interval seconds..."
    sleep $sleep_interval
    waited_time=$((waited_time + sleep_interval))
    if [ $waited_time -ge $max_wait ]; then
        echo "Error: Timed out waiting for available GPU."
        exit 1
    fi
done
