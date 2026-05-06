#!/usr/bin/env bash

if command -v python &> /dev/null; then
    PY_CMD=python
elif command -v python3 &> /dev/null; then
    PY_CMD=python3
else
    echo "Error: Python not found!"
    exit 1
fi

PROJECT_ROOT="${HOME}/repos/lengthgeneralization" # Update this to match your setup
SCRATCH_DIR="/scratch/${USER}" # Update to your cluster's scratch directory

JOB_ID="lengthgen_${CLUSTERId:-0}_${ProcId:-0}_$RANDOM"
JOB_SPECIFIC_DIR="${SCRATCH_DIR}/tmp_jobs/$JOB_ID"
CACHE_DIR="${SCRATCH_DIR}/.cache"

export TORCH_HOME=/tmp/torch_home

export HF_HOME="${CACHE_DIR}/huggingface"
export TORCH_HOME="${CACHE_DIR}/torch"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torch"
export WANDB_DIR="$JOB_SPECIFIC_DIR/wandb"
export WANDB_CACHE_DIR="$JOB_SPECIFIC_DIR/wandb_cache"


mkdir -p $WANDB_DIR
mkdir -p $WANDB_CACHE_DIR
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p "${CACHE_DIR}/wandb"

if [ -f "${HOME}/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat /nethome/okraus/.wandb_api_key)
    echo "WANDB_API_KEY loaded securely from file." >&1
else
    echo "WARNING: ~/.wandb_api_key not found. WandB logging may fail." >&1
fi

# Debugging info
echo "Assigned CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Hostname: $HOSTNAME"
$PY_CMD --version
$PY_CMD -m pip list

#$PY_CMD -m pip install --user -e ${PROJECT_ROOT} --no-deps
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Error: CUDA_VISIBLE_DEVICES is not set. Is this running on a GPU node?"
    NUM_GPUS=1
else
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))
fi

echo "Detected $NUM_GPUS GPUs. Setting tensor-parallel-size to $NUM_GPUS."


# change this as required
TASK="parity"
MODEL_PATH="${SCRATCH_DIR}/lenghtgen/models/sweep_parity_ape_xsmall_ran_pos_delta_len30_wd001_do00_rr03_lr1e-4_tbs256_s46"

$PY_CMD "${PROJECT_ROOT}/src/lengthgen/evaluate_model.py" \
    --task $TASK \
    --model_path $MODEL_PATH \
    --min_len 30 \
    --max_len 50 \
    --out_path "/scratch/okraus/test/new" \
    --num_samples 100 \
    --starting_aid 0 \
    --task_kwargs '{"delta_cot": True}'

echo "Completed evaluation"
