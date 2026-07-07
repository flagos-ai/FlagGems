#!/bin/bash

# Configuration parameters
memory_usage_max=30000     # Maximum memory usage limit (MB)
sleep_time=120             # Wait time (seconds), default is 2 minutes

# Get the number of GPUs
gpu_count=$(mthreads-gmi -L 2>/dev/null | grep -c "GPU ")

if [ "$gpu_count" -eq 0 ]; then
    echo "No Moore Threads GPUs detected. Please ensure you have GPUs installed and properly configured."
    exit 1
fi

echo "Detected $gpu_count Moore Threads GPU(s)."

mthreads-gmi

while true; do
    need_wait=false

    # Check the available memory for each GPU
    for ((i=0; i<$gpu_count; i++)); do
        # Query GPU memory information using mthreads-gmi
        memory_output=$(mthreads-gmi -q -d MEMORY -i $i 2>/dev/null)

        # Parse memory values from "FB Memory Usage" section
        # Format: "Total                                     :  81920MiB"
        memory_total=$(echo "$memory_output" | grep -A 3 "FB Memory Usage" | grep "Total" | grep -oP '\d+' | head -1)
        memory_used=$(echo "$memory_output" | grep -A 3 "FB Memory Usage" | grep "Used" | grep -oP '\d+' | head -1)

        # Check if we got valid memory values
        if [ -z "$memory_used" ] || [ -z "$memory_total" ]; then
            echo "Warning: Failed to query GPU $i memory information."
            continue
        fi

        memory_remin=$((memory_total - memory_used))

        if [ $memory_remin -lt $memory_usage_max ]; then
            need_wait=true
            echo "GPU $i: Used ${memory_used}MB / Total ${memory_total}MB (Available: ${memory_remin}MB < ${memory_usage_max}MB)"
            break
        fi
    done

    if [ "$need_wait" = false ]; then
        echo "All Moore Threads GPUs have sufficient available memory. Proceeding with execution."
        break
    fi

    echo "GPU memory is insufficient, waiting for $sleep_time seconds before retrying..."
    sleep $sleep_time
done
