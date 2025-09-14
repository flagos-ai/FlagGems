TAG=$(date +"%Y_%m_%d_%H_%M")
BASE="$(pwd)"
MODEL="/Change/To/Your/Real/Path/Here/Qwen/Qwen3-8B"                                     # CHANGE THIS to your model path
YAML_CONF="/Change/To/Your/Real/Path/Here/FlagScale/examples/qwen3/conf/serve.yaml"      # CHANGE THIS to your scale conf path

TP=1
GPUS=(0)                                                                                # GPU IDs to use, can be modified as needed

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

INPUT_LEN_LIST="128 512 1024 2048 6144 14336 30720"
OUTPUT_LEN_LIST="128 512 1024 2048"
NUM_PROMPT_LIST="1 100 1000 2000"
MAX_LATENCY_ALLOWED_MS=100000000000

LOG_FOLDER="$BASE/online-benchmark-$PURE_MODEL_NAME/$TAG"
RESULT="$LOG_FOLDER/result.txt"
mkdir -p "$LOG_FOLDER"

echo "result file: $RESULT"

start_vllm_server_on_gpu() {
    local gpu_id=$1
    local port=$2
    local log_file=$3

    pkill -f "vllm.*--port $port"
    echo "start serving GPU ${gpu_id} port ${port}"
    CUDA_VISIBLE_DEVICES=$gpu_id \
    vllm serve $MODEL \
        --disable-log-requests \
        --port $port \
        --no-enable-prefix-caching > "$log_file" 2>&1 &
}

start_scale_server_on_gpu() {
    local gpu_id=$1
    local port=$2

    pkill -f "vllm.*--port $port"
    echo "start serving GPU ${gpu_id} port ${port}"
    export QWEN3_PATH=$MODEL
    export QWEN3_PORT=$port
    flagscale serve qwen3 $YAML_CONF
}

wait_for_server() {
    local port=$1
    for ((i=1; i<=300; i++)); do   # 300*5s=25分钟
        STATUS=$(curl -s -o /dev/null -w "%{http_code}" --noproxy '*' "http://localhost:${port}/v1/models")
        if [[ "$STATUS" -eq 200 ]]; then
            echo "✅ Server on port $port is healthy"
            return 0
        else
            echo "⏳ Server on port $port not ready yet (status=$STATUS, try $i/300)..."
        fi
        sleep 5
    done
    echo "❌ Server on port $port failed to become healthy after timeout"
    return 1
}
run_benchmark_on_gpu() {
    local gpu_id=$1
    local port=$2

    read -r -a input_len_list <<< "$INPUT_LEN_LIST"
    read -r -a output_len_list <<< "$OUTPUT_LEN_LIST"
    read -r -a num_prompt_list <<< "$NUM_PROMPT_LIST"

    for input_len in "${input_len_list[@]}"; do
        for output_len in "${output_len_list[@]}"; do
            for num_prompt in "${num_prompt_list[@]}"; do
                bm_log="$LOG_FOLDER/gpu${gpu_id}_input${input_len}_output${output_len}_num_prompt${num_prompt}.txt"
                echo "GPU $gpu_id: input_len=$input_len output_len=$output_len" > "$bm_log"

                python3 benchmark_serving.py \
                    --backend vllm \
                    --model $MODEL \
                    --dataset-name random \
                    --random-input-len $input_len \
                    --random-output-len $output_len \
                    --ignore-eos \
                    --disable-tqdm \
                    --request-rate inf \
                    --percentile-metrics ttft,tpot,itl,e2el \
                    --goodput e2el:$MAX_LATENCY_ALLOWED_MS \
                    --num-prompts $num_prompt \
                    --port $port &>> "$bm_log"

                throughput=$(grep "Request throughput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
                echo "GPU $gpu_id: num_prompt=$num_prompt throughput=$throughput" >> "$RESULT"
            done
        done
    done
}


port=8004
gpu_id=${GPUS[0]}
log_file="$LOG_FOLDER/gpu${gpu_id}_server.log"

start_scale_server_on_gpu $gpu_id $port
# start_vllm_server_on_gpu  $gpu_id $port $log_file
wait_for_server $port
run_benchmark_on_gpu $gpu_id $port &
wait

echo "✅ DONE, result in $RESULT"
