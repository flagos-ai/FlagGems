#!/bin/bash

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
