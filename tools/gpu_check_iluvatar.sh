#!/bin/bash

# Configuration parameters
memory_usage_max=30000     # Maximum memory usage limit (MB)
sleep_time=120             # Wait time (seconds), default is 2 minutes

# Get the number of GPUs
gpu_count=$(ixsmi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

if [ "$gpu_count" -eq 0 ]; then
    echo "No Iluvatar GPUs detected. Please ensure you have Iluvatar GPUs installed and properly configured."
    exit 1
fi

echo "Detected $gpu_count Iluvatar GPU(s)."

ixsmi

while true; do
    # Query GPU memory usage and total memory
    memory_usage=$(ixsmi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    memory_total=$(ixsmi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)

    # Check if ixsmi command was successful
    if [ $? -ne 0 ]; then
        echo "Failed to query GPU memory information. Please check if ixsmi is working correctly."
        exit 1
    fi

    # Convert query results to arrays
    IFS=$'\n' read -d '' -r -a memory_usage_array <<< "$memory_usage"
    IFS=$'\n' read -d '' -r -a memory_total_array <<< "$memory_total"

    need_wait=false

    # Check the available memory for each GPU
    for ((i=0; i<$gpu_count; i++)); do
        memory_usage_i=${memory_usage_array[$i]}
        memory_total_i=${memory_total_array[$i]}
        memory_remin_i=$((memory_total_i - memory_usage_i))

        if [ $memory_remin_i -lt $memory_usage_max ]; then
            need_wait=true
            echo "GPU $i: Used ${memory_usage_i}MiB / Total ${memory_total_i}MiB (Available: ${memory_remin_i}MiB < ${memory_usage_max}MiB)"
            break
        fi
    done

    if [ "$need_wait" = false ]; then
        echo "All Iluvatar GPUs have sufficient available memory. Proceeding with execution."
        break
    fi

    echo "GPU memory is insufficient, waiting for $sleep_time seconds before retrying..."
    sleep $sleep_time
done
