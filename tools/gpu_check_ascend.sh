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
max_wait=600           # Maximum total wait time (seconds)

# Get the number of NPU chips from npu-smi info output
npu_smi_output=$(npu-smi info 2>&1)

if [ $? -ne 0 ]; then
    echo "Failed to run npu-smi. Please check if npu-smi is installed and working correctly."
    exit 1
fi

npu_count=$(echo "$npu_smi_output" | grep -c "OK")

if [ "$npu_count" -eq 0 ]; then
    echo "No Ascend NPUs detected. Please ensure you have Ascend NPUs installed and properly configured."
    exit 1
fi

echo "Detected $npu_count Ascend NPU chip(s)."
echo "$npu_smi_output"

waited_time=0
while true; do
    npu_smi_output=$(npu-smi info 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "Failed to query NPU information."
        exit 1
    fi

    # Parse HBM-Usage from chip lines
    mapfile -t hbm_lines < <(echo "$npu_smi_output" | grep "0000:" | while IFS= read -r line; do
        echo "$line" | grep -oP '\d+\s*/\s*\d+' | tail -1
    done)

    available_gpus=()
    i=0

    printf " NPU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    for line in "${hbm_lines[@]}"; do
        used_i=$(echo "$line" | awk -F'/' '{gsub(/[[:space:]]/, "", $1); print $1}')
        total_i=$(echo "$line" | awk -F'/' '{gsub(/[[:space:]]/, "", $2); print $2}')

        if [ -z "$used_i" ] || [ -z "$total_i" ]; then
            echo "Warning: Failed to parse memory info for chip $i."
            i=$((i + 1))
            continue
        fi

        free_i=$((total_i - used_i))

        printf "%4d%'13d%'12d%'12d\n" $i ${total_i} ${used_i} ${free_i}
        if [ $free_i -ge $mem_threshold ]; then
            available_gpus+=($i)
        fi
        i=$((i + 1))
    done

    if [ ${#available_gpus[@]} -gt 0 ]; then
        AVAILABLE_GPUS=$(IFS=,; echo "${available_gpus[*]}")
        echo "Available GPUs: ${AVAILABLE_GPUS}"
        break
    fi

    echo "No NPU has sufficient memory, waiting for $sleep_time seconds..."
    sleep $sleep_time
    waited_time=$((waited_time + sleep_time))
    if [ $waited_time -ge $max_wait ]; then
        echo "Error: Timed out waiting for available NPU."
        exit 1
    fi
done
