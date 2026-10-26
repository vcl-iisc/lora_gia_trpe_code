#!/usr/bin/env bash
set -euo pipefail


# There are other paramters that can be set -- check the yaml config files for more customization
# This example runs the benchmark for 10 users after splitting the dataset into 10 parts
# and runs for NUM_TRIALS number of trials.
# ATTACK_NUM_TRIALS dictates how many random initializations are taken for each sample or batch

GPU_ID="${1:-0}"

ATTACK="gias"
BATCH_SIZE=1
STEP_SIZE="${STEP_SIZE:-0.01}"
W_STEP_SIZE="${W_STEP_SIZE:-0.001}"
OBJ_SCALE="${OBJ_SCALE:-1.0}"
TV_SCALE="${TV_SCALE:-1e-3}"
LORA_RANK="${LORA_RANK:-128}"

# FL-related knobs
NUM_TRIALS=3
NUM_QUERIES=1
NUM_LOCAL_UPDATES=4
NUM_DATA_PER_LOCAL_STEP=1
LOCAL_LR=0.0001
NUM_DATA_POINTS=1 # In FedSGD setting, this is the actual BS of federated learning
# In FedAVG setting it determines the total number of data points each user has.
NUM_USERS=1
ATTACK_NUM_TRIALS=1

JOB_NAME="${ATTACK}_multirounds_bs${BATCH_SIZE}_ss${STEP_SIZE}_wss${W_STEP_SIZE}_os${OBJ_SCALE}_tv${TV_SCALE}_rn${LORA_RANK}_q${NUM_QUERIES}_lu${NUM_LOCAL_UPDATES}"

echo "Running ${JOB_NAME} on GPU ${GPU_ID}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python benchmark_breaches.py \
  case/data=CIFAR10 \
  case.model=vit_small \
  case.server.pretrained=True \
  save_reconstruction=true \
  case=1_single_image_small \
  case.data.batch_size=${BATCH_SIZE} \
  case.user.user_type=local_gradient \
  case.user.num_data_points=${NUM_DATA_POINTS} \
  case.server.name=honest_but_curious \
  case.server.num_queries=${NUM_QUERIES} \
  case.peft.type=lora \
  case.peft.lora.rank=${LORA_RANK} \
  case.peft.lora.alpha=$((2 * LORA_RANK)) \
  attack=${ATTACK} \
  dryrun=False \
  attack.restarts.num_trials=${ATTACK_NUM_TRIALS} \
  case.num_users=${NUM_USERS} \
  num_trials=${NUM_TRIALS} \
  attack.objective.type=cosine-similarity \
  attack.objective.posembed_trainable=true \
  attack.objective.apply_block_pos_embed=true \
  attack.optim.step_size=${STEP_SIZE} \
  attack.objective.scale=${OBJ_SCALE} \
  attack.regularization.total_variation.scale=${TV_SCALE} \
  job_name="${JOB_NAME}"

# For Multi-Step set case=4_fedavg_small_scale
# and also use the following
# case.user.local_learning_rate=${LOCAL_LR} \
# case.user.num_local_updates=${NUM_LOCAL_UPDATES} \
# case.user.num_data_per_local_update_step=${NUM_DATA_PER_LOCAL_STEP} \
# case.user.user_type=local_update