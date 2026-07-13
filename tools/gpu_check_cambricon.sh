#!/bin/bash

# Configuration parameters
mem_threshold=30000     # Minimum free memory required (MB)
sleep_interval=120      # Wait time between retries (seconds)
max_wait=600           # Maximum total wait time (seconds)

# Get the number of Cambricon MLU cards from cnmon output
gpu_count=$(cnmon | grep -cP '\d+ MiB/ \d+ MiB')

if [ $? -ne 0 ]; then
    echo "Failed to run cnmon. Please check if cnmon is installed and working correctly."
    exit 1
fi

if [ "$gpu_count" -eq 0 ]; then
    echo "No Cambricon MLUs detected. Please ensure you have MLU cards installed and properly configured."
    exit 1
fi

echo "Detected $gpu_count Cambricon MLU card(s)."

waited_time=0
while true; do
    available_gpus=()
    i=0

    printf " MLU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    while read -r line; do
        used_i=$(echo "$line" | grep -oP '^\d+')
        total_i=$(echo "$line" | grep -oP '\d+(?= MiB$)')

        if [ -z "$used_i" ] || [ -z "$total_i" ]; then
            echo "Warning: Failed to parse memory info for card $i."
            i=$((i + 1))
            continue
        fi

        free_i=$((total_i - used_i))

        printf "%4d%'13d%'12d%'12d\n" $i ${total_i} ${used_i} ${free_i}
        if [ $free_i -ge $mem_threshold ]; then
            available_gpus+=($i)
        fi
        i=$((i + 1))
    done < <(cnmon | grep -oP '\d+ MiB/ \d+ MiB')

    if [ ${#available_gpus[@]} -gt 0 ]; then
        AVAILABLE_GPUS=$(IFS=,; echo "${available_gpus[*]}")
        echo "Available GPUs: ${AVAILABLE_GPUS}"
        break
    fi

    echo "No MLU has sufficient memory, waiting for $sleep_interval seconds..."
    sleep $sleep_interval
    waited_time=$((waited_time + sleep_interval))
    if [ $waited_time -ge $max_wait ]; then
        echo "Error: Timed out waiting for available MLU."
        exit 1
    fi
done
