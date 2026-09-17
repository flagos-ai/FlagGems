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

gpu_count=$(mx-smi | awk '/Attached/ {print $4}' 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "Failed to run mx-smi. Please check if mx-smi is installed and working correctly."
    exit 1
fi

if [ "$gpu_count" -eq 0 ]; then
    echo "No MetaX GPUs detected. Please ensure you have MetaX GPUs installed and properly configured."
    exit 1
fi

echo "Detected $gpu_count MetaX GPU chip(s)."

waited_time=0
while true; do
    memory_info=$(mx-smi | awk '/MiB[[:space:]]| A/ { print $9 }')
    readarray -t lines <<< "$memory_info"
    available_gpus=()
    i=0

    printf " GPU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    for line in "${lines[@]}"; do
        used_i=$(echo "$line" | awk -F'/' '{print $1}')
        total_i=$(echo "$line" | awk -F'/' '{ print $2}')

        if [ -z "$used_i" ] || [ -z "$total_i" ]; then
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

    echo "No GPU has sufficient memory, waiting for $sleep_time seconds..."
    sleep $sleep_time
    waited_time=$((waited_time + sleep_time))
    if [ $waited_time -ge $max_wait ]; then
        echo "Error: Timed out waiting for available GPU."
        exit 1
    fi
done
