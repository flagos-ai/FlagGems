#!/bin/bash

# Configuration parameters
mem_threshold=30000     # Maximum memory usage limit (MB)
sleep_time=120             # Wait time (seconds), default is 2 minutes

export KUIPER_HOME=/home/secure/runtime/kuiper
export PATH=$KUIPER_HOME/bin:$PATH
export LD_LIBRARY_PATH=$KUIPER_HOME/lib:$LD_LIBRARY_PATH

# Get the number of GPUs
gpu_count=$(tsm_smi 2>/dev/null | awk '/Chip count:/ { print $6 }')

if [ "$gpu_count" -eq 0 ]; then
    echo "No TsingMicro cards detected. Please ensure you have TsingMicro cards installed and properly configured."
    exit 1
fi

echo "Detected $gpu_count TsingMicro card(s)."

while true; do
    memory_usage=$(tsm_smi | awk '{if ($2=="|") {print $9} else if ($4=="|" && $11!="|") {print $11} }')
    memory_total=$(tsm_smi | awk '{if ($2=="|") {print $11} else if ($4=="|" && $11!="|") {print $13} }')

    # Check if tsm_smi command was successful
    if [ $? -ne 0 ]; then
        echo "Failed to query GPU memory information. Please check if tsm_smi is working correctly."
        exit 1
    fi

    # Convert query results to arrays
    IFS=$'\n' read -d '' -r -a usage_array <<< "$memory_usage"
    IFS=$'\n' read -d '' -r -a total_array <<< "$memory_total"

    need_wait=false

    # Check the available memory for each GPU
    printf " GPU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    for ((i=0; i<$gpu_count; i++)); do
        # trim 'M' from number
        total_i=${total_array[$i]%?}
        used_i=${usage_array[$i]%?}
        free_i=$((total_i - used_i))

        printf "%4d%'13d%'12d%'12d\n" $i ${total_i} ${used_i} ${free_i}
        if [ $free_i -lt $mem_threshold ]; then
            need_wait=true
            break
        fi
    done

    if [ "$need_wait" = false ]; then
        echo "All GPUs have sufficient memory."
        break
    fi

    echo "GPU memory is insufficient, waiting for $sleep_time seconds before retrying..."
    sleep $sleep_time
done
