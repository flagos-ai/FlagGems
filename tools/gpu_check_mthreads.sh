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

export MUSA_INSTALL_PATH=/usr/local/musa
export PATH=$MUSA_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MUSA_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Get the number of GPUs
gpu_count=$(mthreads-gmi -L 2>/dev/null | grep -c "GPU ")

if [ "$gpu_count" -eq 0 ]; then
    echo "No Moore Threads GPUs detected. Please ensure you have GPUs installed and properly configured."
    exit 1
fi
echo "Detected $gpu_count Moore Threads GPU(s)."

waited_time=0
while true; do
    available_gpus=()

    printf " GPU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    for ((i=0; i<$gpu_count; i++)); do
        memory_output=$(mthreads-gmi -q -d MEMORY -i $i 2>/dev/null)

        total_i=$(echo "$memory_output" | grep -A 3 "FB Memory Usage" | grep "Total" | grep -oP '\d+' | head -1)
        used_i=$(echo "$memory_output" | grep -A 3 "FB Memory Usage" | grep "Used" | grep -oP '\d+' | head -1)

        if [ -z "$used_i" ] || [ -z "$total_i" ]; then
            echo "Warning: Failed to query GPU $i memory information."
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
