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

TAG=$(date +"%Y_%m_%d_%H_%M")
BASE="$(pwd)"
MODEL="/home/tianjinjin/.cache/modelscope/hub/models/Qwen/Qwen3-8B"  # CHANGE THIS to your model path
PURE_MODEL_NAME=$(basename "$MODEL")
LOG_FOLDER="$BASE/offline-benchmark-$PURE_MODEL_NAME/$TAG"
mkdir -p "$LOG_FOLDER"

INPUT_LEN_LIST="128 512 1024 2048 6144 14336 30720"
OUTPUT_LEN_LIST="128 512 1024 2048"
NUM_PROMPT_LIST="1 100 1000 2000"

read -r -a input_len_list <<< "$INPUT_LEN_LIST"
read -r -a output_len_list <<< "$OUTPUT_LEN_LIST"
read -r -a num_prompt_list <<< "$NUM_PROMPT_LIST"

for input_len in "${input_len_list[@]}"; do
    for output_len in "${output_len_list[@]}"; do
        for num_prompt in "${num_prompt_list[@]}"; do
            bm_log="$LOG_FOLDER/input${input_len}_output${output_len}_num_prompt${num_prompt}.txt"
            python benchmark_throughput.py --model $MODEL \
            --dataset-name random \
            --input-len ${input_len} \
            --output-len ${output_len} \
            --num-prompts ${num_prompt} \
            --no-enable-prefix-caching &>> "$bm_log"
        done
    done
done
