#!/bin/bash
# -----------------------------------------------------------------------------
# Master Grid Search Launcher
# -----------------------------------------------------------------------------
# Submits GPU jobs for each hyperparameter combination.
# This job itself runs on a low-priority CPU partition.
#
# Usage:
#   sbatch run_grid.slurm
# -----------------------------------------------------------------------------

set -euo pipefail

# Usage:
#   ./run_multirounds_sweep_april.sh [GPU_ID]
# Example:
#   ./run_multirounds_sweep_april.sh 0

START_GPU_ID="${1:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"
PARALLEL_GPUS="${PARALLEL_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
running=0
count=0

ATTACK="${ATTACK:-gias}"
NUM_TRIALS="${NUM_TRIALS:-10}"
NUM_QUERIES="${NUM_QUERIES:-1}"

STEP_SIZES=(0.1)
W_STEP_SIZES=(0.001)
OBJ_SCALES=(0.004)
# PE_SCALES=(0.1 0.01 0.001)
TV_SCALES=(0.0001)
# IMG_PRIOR_SCALES=(0.2)
# PATCH_PRIOR_SCALES=(0.0001)
L2_REG_SCALES=(0.0001)
GROUP_REG_SCALES=(0.0001)
LORA_RANKS=(16 32 64 128)
NUM_DATA_POINTS=(16)
NUM_EPOCHS=(1 2)
NUM_DATA_PER_LOCAL_STEP=(4 8)
LOCAL_LRS=(0.0001)
NUM_USERS=(1)
ATTACK_NUM_TRIALS=(3)

for STEP_SIZE in "${STEP_SIZES[@]}"; do
for W_STEP_SIZE in "${W_STEP_SIZES[@]}"; do
for OBJ_SCALE in "${OBJ_SCALES[@]}"; do
# for PE_SCALE in "${PE_SCALES[@]}"; do
for TV_SCALE in "${TV_SCALES[@]}"; do
# for IMG_PRIOR_SCALE in "${IMG_PRIOR_SCALES[@]}"; do
# for PATCH_PRIOR_SCALE in "${PATCH_PRIOR_SCALES[@]}"; do
for L2_REG_SCALE in "${L2_REG_SCALES[@]}"; do
for GROUP_REG_SCALE in "${GROUP_REG_SCALES[@]}"; do
for LORA_RANK in "${LORA_RANKS[@]}"; do
for EPOCHS in "${NUM_EPOCHS[@]}"; do
for DPS in "${NUM_DATA_PER_LOCAL_STEP[@]}"; do
for LLR in "${LOCAL_LRS[@]}"; do
for NDP in "${NUM_DATA_POINTS[@]}"; do
for NU in "${NUM_USERS[@]}"; do
for ATRIALS in "${ATTACK_NUM_TRIALS[@]}"; do
parallel_gpu_id=$(( (START_GPU_ID + count) % PARALLEL_GPUS ))
NUM_LOCAL_STEPS=$(( (NDP / DPS) * EPOCHS ))
echo "Running sweep item: ss=${STEP_SIZE}, os=${OBJ_SCALE}, tv=${TV_SCALE}, l2=${L2_REG_SCALE}, gr=${GROUP_REG_SCALE}, rn=${LORA_RANK}, epochs=${EPOCHS}, local_steps=${NUM_LOCAL_STEPS}, dps=${DPS}, llr=${LLR}, ndp=${NDP}, nu=${NU}, atrials=${ATRIALS} on GPU ${parallel_gpu_id}"
(
    CUDA_VISIBLE_DEVICES="${parallel_gpu_id}" python benchmark_breaches_multirounds.py \
    seed=42 \
    case=4_fedavg_small_scale \
    case/data=CIFAR10 \
    case.model.name="vit_small" \
    case.model.patch_size=16 \
    save_reconstruction=true \
    case.data.batch_size=1 \
    case.user.user_type=local_update \
    case.user.num_data_points="${NDP}" \
    case.user.num_local_updates="${NUM_LOCAL_STEPS}" \
    case.user.num_data_per_local_update_step="${DPS}" \
    case.user.local_learning_rate="${LLR}" \
    case.server.name=honest_but_curious_with_memory \
    case.server.num_queries="${NUM_QUERIES}" \
    case.server.pretrained=true \
    case.server.pretrained_path=null \
    case.peft.type=lora \
    case.peft.lora.rank="${LORA_RANK}" \
    case.peft.lora.alpha=$((2 * LORA_RANK)) \
    case.peft.lora.initialisation="uniform" \
    attack="${ATTACK}" \
    dryrun=False \
    attack.restarts.num_trials="${ATRIALS}" \
    case.num_users="${NU}" \
    num_trials="${NUM_TRIALS}" \
    attack.objective.type=cosine-similarity \
    attack.objective.posembed_trainable=false \
    attack.objective.apply_block_pos_embed=false \
    attack.optim.step_size="${STEP_SIZE}" \
    attack.optim_w.step_size="${W_STEP_SIZE}" \
    attack.objective.scale="${OBJ_SCALE}" \
    attack.regularization.total_variation.scale="${TV_SCALE}" \
    attack.regularization.l2_regularization.scale="${L2_REG_SCALE}" \
    attack.regularization.group_lazy_regularization.scale="${GROUP_REG_SCALE}" \
    job_name="fedavg_${ATTACK}_ss${STEP_SIZE}_os${OBJ_SCALE}_tv${TV_SCALE}_l2${L2_REG_SCALE}_gr${GROUP_REG_SCALE}_rn${LORA_RANK}_q1_lu${NUM_LOCAL_STEPS}_ndpls${DPS}"
) &

running=$((running + 1))
count=$((count + 1))

if [[ "${running}" -ge "${MAX_PARALLEL}" ]]; then
    wait
    running=0
fi
done
done
done
done
done
done
done
done
done
done
done
done
done
# done
# done
# done
wait

# attack.regularization.image_prior.scale="${IMG_PRIOR_SCALE}" \
# attack.regularization.patch_prior.scale="${PATCH_PRIOR_SCALE}" \