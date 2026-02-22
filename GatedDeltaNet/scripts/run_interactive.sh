OUTPUT_ROOT="/lustre/fsw/portfolios/nvr/users/ahatamizadeh/code/GatedDeltaNet/lit_gpt"

EVAL_ITERS=25 # number of evaluation iterations

NAME="interactive_test4"
TRAIN_DATA=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_nvgptvision/datasets/llm_next_gen/data/SlimPajama-627B_pretok/train
VALIDATION_DATA=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_nvgptvision/datasets/llm_next_gen/data/SlimPajama-627B_pretok/validation
SAVE_DIR="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_nvgptvision/checkpoints/llm_next_gen/ah"
MODEL='GatedDeltaNet_1.3B'
MODEL='GatedDeltaNet_0.4B'
MODEL='GatedDeltaNet_H1_0.4B'
MODEL='GatedDeltaNet_H1_1.3B'
CONFIG='tsz512x4k_20B'
LR=1e-3
MICRO_BATCH_SIZE=2

TRI_CACHE_DIR="${SAVE_DIR}/triton/${NAME}/"
export TRITON_HOME="/lustre/fsw/portfolios/nvr/users/ahatamizadeh/.triton2/interactive2/"
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/nvr/users/ahatamizadeh/.triton2/interactive2/"

python ../pretrain.py \
--train_data_dir ${TRAIN_DATA} \
--val_data_dir ${VALIDATION_DATA} \
--output_root ${SAVE_DIR} \
--exp_name ${NAME} \
--model_name ${MODEL} \
--train_config ${CONFIG} \
--eval_iters ${EVAL_ITERS} \
--learning_rate ${LR} \
--interactive_job \
--micro_batch_size ${MICRO_BATCH_SIZE} \
--debug \
