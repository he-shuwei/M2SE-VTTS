#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

CKPT=""
OUTDIR=""
BATCH_SIZE=""
TEST_SET="test_seen"
NUM_GPUS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)       CKPT="$2";       shift 2 ;;
        --outdir)     OUTDIR="$2";     shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --test_set)   TEST_SET="$2";   shift 2 ;;
        --num_gpus)   NUM_GPUS="$2";   shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "${CKPT}" ] || [ -z "${OUTDIR}" ] || [ -z "${BATCH_SIZE}" ]; then
    echo "Usage: $0 --ckpt CKPT_PATH --outdir OUTPUT_DIR --batch_size BATCH_SIZE [--test_set TEST_SET] [--num_gpus N]"
    exit 1
fi

[[ "${CKPT}" = /* ]] || CKPT="${PROJECT_ROOT}/${CKPT}"
[[ "${OUTDIR}" = /* ]] || OUTDIR="${PROJECT_ROOT}/${OUTDIR}"

if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}" >&2
    exit 1
fi

CKPT_DIR="$(dirname "${CKPT}")"
if [ -f "${CKPT_DIR}/config.yaml" ]; then
    CONFIG="${CKPT_DIR}/config.yaml"
else
    CONFIG="${PROJECT_ROOT}/configs/m2se_vtts.yaml"
fi

EXP_NAME="$(basename "${CKPT_DIR}")"

if [ -z "${NUM_GPUS}" ]; then
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
fi

if [ -f "${PROJECT_ROOT}/env/m2se-vtts/bin/python" ]; then
    PYTHON="${PROJECT_ROOT}/env/m2se-vtts/bin/python"
else
    PYTHON="python3"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
unset CUDA_VISIBLE_DEVICES

cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs/infer"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_${TEST_SET}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================="
echo "  M2SE-VTTS Inference"
echo "  Checkpoint  : ${CKPT}"
echo "  Config      : ${CONFIG}"
echo "  Exp name    : ${EXP_NAME}"
echo "  Test set    : ${TEST_SET}"
echo "  Output      : ${OUTDIR}"
echo "  Batch size  : ${BATCH_SIZE}"
echo "  GPUs        : ${NUM_GPUS}"
echo "  Log         : ${LOG_FILE}"
echo "============================================="

nohup ${PYTHON} scripts/infer/infer.py \
    --config "${CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --load_ckpt "${CKPT}" \
    --output_dir "${OUTDIR}" \
    --test_set "${TEST_SET}" \
    --num_gpus "${NUM_GPUS}" \
    --batch_size "${BATCH_SIZE}" \
    --use_gt_dur \
    --use_gt_f0 \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    > "${LOG_FILE}" 2>&1 &

echo ""
echo "  Inference started in background (PID: $!)"
echo "  Monitor:  tail -f ${LOG_FILE}"
echo "  Kill:     bash scripts/infer/kill_infer.sh"
echo ""
