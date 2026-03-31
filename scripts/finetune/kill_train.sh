#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
EXP_NAME="${1:-m2se_vtts}"
PID_FILE="${LOG_DIR}/${EXP_NAME}.pid"

echo "============================================="
echo "  Stop M2SE-VTTS Training: ${EXP_NAME}"
echo "============================================="

if [ -f "${PID_FILE}" ]; then
    MAIN_PID=$(cat "${PID_FILE}")
    if kill -0 "${MAIN_PID}" 2>/dev/null; then
        echo "[INFO] Killing main process PID=${MAIN_PID} and its process tree..."
        PGID=$(ps -o pgid= -p "${MAIN_PID}" 2>/dev/null | tr -d ' ')
        if [ -n "${PGID}" ] && [ "${PGID}" != "0" ]; then
            kill -TERM -"${PGID}" 2>/dev/null || true
            sleep 3
            kill -9 -"${PGID}" 2>/dev/null || true
        else
            kill "${MAIN_PID}" 2>/dev/null || true
        fi
        echo "[OK] Process group killed"
    else
        echo "[INFO] PID=${MAIN_PID} no longer exists"
    fi
    rm -f "${PID_FILE}"
else
    echo "[WARN] PID file not found: ${PID_FILE}"
fi

REMAINING=$(pgrep -f "m2se_vtts.run.*--exp_name ${EXP_NAME}" 2>/dev/null || true)
if [ -n "${REMAINING}" ]; then
    echo "[INFO] Remaining main processes: ${REMAINING}"
    echo "${REMAINING}" | xargs kill -9 2>/dev/null || true
fi

SPAWN_PIDS=$(pgrep -f "spawn_main.*tracker_fd" 2>/dev/null || true)
if [ -n "${SPAWN_PIDS}" ]; then
    echo "[INFO] Remaining spawn child processes: $(echo ${SPAWN_PIDS} | wc -w)"
    echo "${SPAWN_PIDS}" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

echo "[DONE] Training stopped"
