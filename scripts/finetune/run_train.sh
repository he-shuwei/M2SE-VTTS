#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG="${1:-configs/m2se_vtts.yaml}"
EXP_NAME="${2:-m2se_vtts}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

if [ -f "${PROJECT_ROOT}/env/m2se-vtts/bin/python" ]; then
    PYTHON="${PROJECT_ROOT}/env/m2se-vtts/bin/python"
else
    PYTHON="python3"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
if [ -z "${MASTER_PORT:-}" ]; then
    export MASTER_PORT=$(${PYTHON} -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
fi

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

if [ ! -f "${CONFIG}" ]; then
    echo "[ERROR] Config file not found: ${CONFIG}" >&2
    exit 1
fi

NUM_GPUS=$(${PYTHON} -c "import torch; print(torch.cuda.device_count())")
echo "============================================="
echo "  M2SE-VTTS Finetune"
echo "  Config:   ${CONFIG}"
echo "  Exp:      ${EXP_NAME}"
echo "  GPUs:     ${NUM_GPUS}"
echo "  Log:      ${LOG_FILE}"
echo "  PID file: ${LOG_DIR}/${EXP_NAME}.pid"
echo "============================================="

nohup ${PYTHON} -m m2se_vtts.run \
    --config "${CONFIG}" \
    --exp_name "${EXP_NAME}" \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${LOG_DIR}/${EXP_NAME}.pid"

echo "[OK] Training started  PID=${TRAIN_PID}"
echo "     View log:  tail -f ${LOG_FILE}"
echo "     Stop:      bash scripts/finetune/kill_train.sh ${EXP_NAME}"
