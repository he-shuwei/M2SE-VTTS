#!/usr/bin/env bash

echo "Killing inference processes..."

MAIN_PIDS=$(pgrep -f "scripts/infer/infer.py" 2>/dev/null)
if [ -n "${MAIN_PIDS}" ]; then
    for PID in ${MAIN_PIDS}; do
        PGID=$(ps -o pgid= -p ${PID} 2>/dev/null | tr -d ' ')
        if [ -n "${PGID}" ]; then
            kill -9 -${PGID} 2>/dev/null
            echo "  Killed process group ${PGID} (main PID ${PID})"
        else
            kill -9 ${PID} 2>/dev/null
            echo "  Killed PID ${PID}"
        fi
    done
else
    echo "  No infer.py main processes found"
fi

pkill -9 -f "multiprocessing.spawn" 2>/dev/null && echo "  Killed orphan spawn workers"
pkill -9 -f "multiprocessing.forkserver" 2>/dev/null

echo "Done."
