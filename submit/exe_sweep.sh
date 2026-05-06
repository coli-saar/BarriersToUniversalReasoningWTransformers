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

# Expose the src directory to Python so all modules are importable
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

export HF_HOME="${CACHE_DIR}/huggingface"
export TORCH_HOME="${CACHE_DIR}/torch"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torch"
export WANDB_DIR="$JOB_SPECIFIC_DIR/wandb"
export WANDB_CACHE_DIR="$JOB_SPECIFIC_DIR/wandb_cache"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"
mkdir -p "${CACHE_DIR}/wandb"

if [ -f "${HOME}/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
    echo "WANDB_API_KEY loaded." >&1
else
    echo "WARNING: ~/.wandb_api_key not found. WandB logging may fail." >&1
fi

echo "Hostname: $HOSTNAME"
$PY_CMD --version

if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    echo "WARNING: nvidia-smi not found. Assuming 0 GPUs."
    NUM_GPUS=0
fi

#unset CUDA_VISIBLE_DEVICES
echo "Detected $NUM_GPUS GPUs."

CONFIG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Usage: $0 --config path/to/sweep.yaml"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

echo "Using config: $CONFIG"

$PY_CMD "${PROJECT_ROOT}/src/lengthgen/launch.py" \
    --config "$CONFIG" \
    --num_gpus "$NUM_GPUS"

echo "All experiments completed!"