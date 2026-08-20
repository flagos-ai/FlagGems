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

# Check if nvidia-smi exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi command not found."
    exit 1
fi

# Get the number of GPUs
gpu_count=$(nvidia-smi -L 2>/dev/null | grep -c "NVIDIA ")

if [ "$gpu_count" -eq 0 ]; then
    echo "No NVIDIA GPU cards detected."
    exit 1
fi

echo "Detected $gpu_count NVIDIA GPU card(s)."

waited_time=0
while true; do
    available_gpus=()

    printf " GPU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    for ((i=0; i<$gpu_count; i++)); do
        mem_line=$(nvidia-smi -i $i 2>/dev/null | grep -oP '\d+MiB\s*/\s*\d+MiB')

        if [ -z "$mem_line" ]; then
            echo "Warning: Failed to query memory on GPU $i."
            continue
        fi

        used_i=$(echo "$mem_line" | grep -oP '^\d+')
        total_i=$(echo "$mem_line" | grep -oP '/\s*\K\d+')

        if [ -z "$total_i" ] || [ -z "$used_i" ]; then
             echo "Warning: Parse error for GPU $i. Raw: '$mem_line'"
             continue
        fi

        free_i=$((total_i - used_i))

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

    echo "No GPU has sufficient memory, waiting for $sleep_time seconds..."
    sleep $sleep_time
    waited_time=$((waited_time + sleep_time))
    if [ $waited_time -ge $max_wait ]; then
        echo "Error: Timed out waiting for available GPU."
        exit 1
    fi
done
