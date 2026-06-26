#!/bin/bash

# Configuration parameters
mem_threshold=30000     # Maximum memory usage limit (MB)
sleep_interval=120      # Wait time (seconds), default is 2 minutes
max_wait=1200           # Maximum wait time (seconds), default is 20 minutes

# Get the number of Cambricon MLU cards from cnmon output
# Memory lines look like:
# | 7         52C     85 W/ 500 W |     0 MiB/ 81920 MiB | FULL          Default |
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
    need_wait=false
    i=0

    printf " MLU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    cnmon | grep -oP '\d+ MiB/ \d+ MiB' | while read -r line; do
        used_i=$(echo "$line" | grep -oP '^\d+')
        total_i=$(echo "$line" | grep -oP '\d+(?= MiB$)')

        if [ -z "$used_i" ] || [ -z "$total_i" ]; then
            echo "Warning: Failed to parse memory info for card $i."
            i=$((i + 1))
            continue
        fi

        free_i=$((total_i - used_i))

        printf "%4d%'13d%'12d%'12d\n" $i ${total_i} ${used_i} ${free_i}
        if [ $free_i -lt $mem_threshold ]; then
            need_wait=true
            break
        fi
        i=$((i + 1))
    done

    if [ "$need_wait" = false ]; then
        echo "All MLUs have sufficient memory, proceeding with execution."
        break
    fi

    echo "MLU memory is insufficient, waiting for $sleep_interval seconds before retrying..."
    sleep $sleep_interval
    waited_time=$(( waited_time + sleep_interval ))
    if [ $waited_time -gt $max_wait ]; then
        break
    fi
done
