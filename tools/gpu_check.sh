#!/bin/bash

# Function to display usage information
usage() {
    echo "Monitor GPU memory usage and wait until sufficient memory is available before proceeding."
    echo
    echo "This script checks the available memory on specified NVIDIA GPUs. If the available memory"
    echo "on any specified GPU is below the specified memory usage limit, the script will wait for "
    echo "a specified time and retry."
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -m, --memory <MB>     Set the maximum memory usage limit (default: $DEFAULT_MEMORY_USAGE_MAX MB)."
    echo "                        This is the minimum amount of free memory required on each GPU."
    echo "  -s, --sleep <seconds> Set the wait time between checks (default: $DEFAULT_SLEEP_TIME seconds)."
    echo "                        This is the time the script will wait before rechecking GPU memory."
    echo "  -g, --gpu <ids>       Set the GPU IDs to monitor (default: $DEFAULT_GPU_IDS)"
    echo "                        Use 'all' to monitor all GPUs, or specify a comma-separated list (e.g., '0,1')."
    echo "  -q, --quiet           Enable quiet mode (default: false)"
    echo "  -h, --help            Display this help message."
    echo
    echo "Examples:"
    echo "  $0                           # Run with default values (30000 MB memory limit, 120 seconds sleep)"
    echo "  $0 --memory 20000            # Set memory limit to 20000 MB"
    echo "  $0 --sleep 60                # Set sleep time to 60 seconds"
    echo "  $0 --memory 15000 --sleep 30 # Set memory limit to 15000 MB and sleep time to 30 seconds"
    echo "  $0 --memory 15000 --gpu 0,3  # Set memory limit to 15000 MB and monitor GPU 0 and GPU 3"
    echo "  $0 --quiet                   # Enable quiet mode"
    echo
    echo "Note: Ensure that nvidia-smi is installed and properly configured to use this script."
    exit 1
}

# Check if nvidia-smi is installed and working
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi is not installed or not in your \$PATH."
    echo "Please install NVIDIA drivers and ensure nvidia-smi is available."
    exit 1
fi

# Try running nvidia-smi to check if it works
if ! nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi is installed but failed to run."
    echo "Please check if NVIDIA drivers are properly configured."
    exit 1
fi

# Configuration parameters
DEFAULT_MEMORY_USAGE_MAX=30000  # Default maximum memory usage limit (MB)
DEFAULT_SLEEP_TIME=120          # Default wait time (seconds), default is 2 minutes
DEFAULT_GPU_IDS="all"           # Default GPU IDs to monitor ("all" or comma-separated list, e.g., "0,1")
DEFAULT_QUIET_MODE=false        # Default not quiet mode

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--memory)
            memory_usage_max="$2"
            shift
            ;;
        -s|--sleep)
            sleep_time="$2"
            shift
            ;;
        -g|--gpu)
            gpu_ids="$2"
            shift
            ;;
        -q|--quiet)
            DEFAULT_QUIET_MODE=true
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
    shift
done

# Set default values if not provided
memory_usage_max=${memory_usage_max:-$DEFAULT_MEMORY_USAGE_MAX}
sleep_time=${sleep_time:-$DEFAULT_SLEEP_TIME}
gpu_ids=${gpu_ids:-$DEFAULT_GPU_IDS}

# Get the number of GPUs
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ "$gpu_count" -eq 0 ]; then
    echo "[Error]: No GPUs detected. Please ensure you have NVIDIA GPUs installed and properly configured."
    exit 1
fi

echo "[INFO]: Detected $gpu_count GPUs."

# Parse GPU IDs
if [ "$gpu_ids" == "all" ]; then
    gpu_ids=$(seq -s ',' 0 $((gpu_count - 1)))
fi

if [ "$DEFAULT_QUIET_MODE" = false ]; then
    nvidia-smi
fi

# Clean up GPU IDs (remove spaces and replace other delimiters with commas)
gpu_ids=$(echo "$gpu_ids" | tr -d ' ' | tr ';' ',')
echo "[INFO]: Monitoring GPU ID(s): $gpu_ids"

# Split the comma-separated string into an array
IFS=',' read -r -a gpu_id_array <<< "$gpu_ids"

# Flag that indicate the first time to check if memory_usage_max is greater than any GPU's total memory
first_time_check=true

while true; do
    # Query GPU memory usage and total memory
    memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)

    # Check if nvidia-smi command was successful
    if [ $? -ne 0 ]; then
        echo "[Error]: Failed to query GPU memory information. Please check if nvidia-smi is working correctly."
        exit 1
    fi

    # Convert query results to arrays
    IFS=$'\n' read -d '' -r -a memory_usage_array <<< "$memory_usage"
    IFS=$'\n' read -d '' -r -a memory_total_array <<< "$memory_total"

    # Check if memory_usage_max is greater than any GPU's total memory
    if [ "$first_time_check" = true ]; then
        first_time_check=false
        for gpu_id in "${gpu_id_array[@]}"; do
            memory_total_i=${memory_total_array[$gpu_id]}
            if [ "$memory_usage_max" -gt "$memory_total_i" ]; then
                echo "[Error]: memory_usage_max ($memory_usage_max MB) is greater than GPU $gpu_id's total memory ($memory_total_i MB)."
                echo "[Error]: Please set memory_usage_max to a value less than or equal to the GPU's total memory."
                exit 1
            fi
        done
    fi

    need_wait=false

    # Check the available memory for each GPU
    for i in "${gpu_id_array[@]}"; do
        memory_usage_i=${memory_usage_array[$i]}
        memory_total_i=${memory_total_array[$i]}
        memory_remin_i=$((memory_total_i - memory_usage_i))

        if [ $memory_remin_i -lt $memory_usage_max ]; then
            echo "[WAIT]: GPU $i has insufficient available memory: $memory_remin_i MB (required: $memory_usage_max MB)"
            need_wait=true
            break
        fi
    done

    if [ "$need_wait" = false ]; then
        echo "[INFO]: All specified GPUs have sufficient available memory. Proceeding with execution."
        break
    fi

    message="GPU memory is insufficient, waiting for $sleep_time seconds before retrying..."
    echo "[INFO]: $message"

    # Write a deviding line
    spaces=$(printf ' %.0s' {1..8})
    dashes=$(printf -- '-%.0s' $(seq 1 $((${#message}))))
    line="$spaces$dashes"
    echo "$line"
    sleep $sleep_time
done
