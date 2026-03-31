#!/bin/bash

set -e

JOBS=64
SKIP_PREPARE=false
BACKGROUND=false
PROCESSED_DIR="data/processed_data"
MFA_INPUT_DIR="data/processed_data/mfa/inputs"
MFA_OUTPUT_DIR="data/processed_data/mfa/outputs"
MFA_DICT_DIR="data/processed_data/mfa/dict"
LOG_FILE="logs/mfa_align.log"

MFA_CONDA_ENV="${MFA_CONDA_ENV:-$(conda info --base)/envs/mfa}"
export PATH="$MFA_CONDA_ENV/bin:$PATH"
MFA_BIN="$MFA_CONDA_ENV/bin/mfa"

MFA_MODELS_DIR="${MFA_MODELS_DIR:-${HOME}/Documents/MFA/pretrained_models}"
DICT_PATH="$MFA_MODELS_DIR/dictionary/english_us_arpa.dict"
ACOUSTIC_PATH="$MFA_MODELS_DIR/acoustic/english_us_arpa.zip"
G2P_PATH="$MFA_MODELS_DIR/g2p/english_us_arpa.zip"

MERGED_DICT="$MFA_DICT_DIR/english_us_arpa_full.dict"

BEAM=100
RETRY_BEAM=400

while [[ $# -gt 0 ]]; do
  case $1 in
    --jobs=*) JOBS="${1#*=}"; shift ;;
    --jobs)   JOBS="$2"; shift 2 ;;
    --skip-prepare) SKIP_PREPARE=true; shift ;;
    --bg)     BACKGROUND=true; shift ;;
    *)        shift ;;
  esac
done

if [ "$BACKGROUND" = true ]; then
    mkdir -p logs
    ARGS="${@/--bg/}"
    nohup bash "$0" $ARGS > "$LOG_FILE" 2>&1 &
    BG_PID=$!
    echo "=============================="
    echo " MFA started in background"
    echo "=============================="
    echo "  PID:  $BG_PID"
    echo "  Log:  $LOG_FILE"
    echo ""
    echo "  Monitor: tail -f $LOG_FILE"
    echo "  Status:  kill -0 $BG_PID && echo running || echo done"
    echo "  Stop:    kill $BG_PID"
    exit 0
fi

echo "=============================="
echo " M2SE-VTTS MFA Alignment"
echo "=============================="
echo "  Jobs:        $JOBS"
echo "  Beam:        $BEAM / $RETRY_BEAM"
echo "  MFA bin:     $MFA_BIN"
echo "  Input:       $MFA_INPUT_DIR"
echo "  Output:      $MFA_OUTPUT_DIR"
echo ""

if [ "$SKIP_PREPARE" = false ]; then
    echo "[Step 1] Preparing MFA inputs..."
    python scripts/data_preprocessing/finetune/04_prepare_mfa.py \
        --processed_dir $PROCESSED_DIR \
        --output_dir $MFA_INPUT_DIR
else
    echo "[Step 1] Skipping prepare (--skip-prepare)"
fi

N_WAV=$(find $MFA_INPUT_DIR -name '*.wav' | wc -l)
N_SPK=$(ls $MFA_INPUT_DIR | wc -l)
echo "  Input items:    $N_WAV"
echo "  Input speakers: $N_SPK"

echo ""
echo "[Step 2] Generating pronunciations for OOV words..."

OOV_DICT="$MFA_DICT_DIR/oov_generated.dict"
mkdir -p "$MFA_DICT_DIR"

$MFA_BIN g2p \
    $MFA_INPUT_DIR \
    $G2P_PATH \
    $OOV_DICT \
    --dictionary_path $DICT_PATH \
    --num_pronunciations 1 \
    --num_jobs $JOBS

OOV_COUNT=$(wc -l < "$OOV_DICT" 2>/dev/null || echo 0)
echo "  OOV words with generated pronunciations: $OOV_COUNT"

cat "$DICT_PATH" "$OOV_DICT" > "$MERGED_DICT"
TOTAL_WORDS=$(wc -l < "$MERGED_DICT")
echo "  Merged dictionary: $TOTAL_WORDS entries → $MERGED_DICT"

echo ""
echo "[Step 3] Running MFA alignment..."
echo "  --jobs $JOBS, --beam $BEAM, --retry_beam $RETRY_BEAM"
echo "  Dictionary: $MERGED_DICT"
mkdir -p $MFA_OUTPUT_DIR
mkdir -p logs

$MFA_BIN align \
    $MFA_INPUT_DIR \
    $MERGED_DICT \
    $ACOUSTIC_PATH \
    $MFA_OUTPUT_DIR \
    --jobs $JOBS \
    --beam $BEAM \
    --retry_beam $RETRY_BEAM \
    --uses_speaker_adaptation false \
    --clean \
    --overwrite \
    2>&1 | tee $LOG_FILE

N_TG=$(find $MFA_OUTPUT_DIR -name '*.TextGrid' | wc -l)
echo ""
echo "=============================="
echo " MFA Alignment Results"
echo "=============================="
echo "  TextGrid files: $N_TG / $N_WAV"
if [ $N_WAV -gt 0 ]; then
    echo "  Success rate:   $(echo "scale=1; $N_TG * 100 / $N_WAV" | bc)%"
fi
echo "  Output dir:     $MFA_OUTPUT_DIR"
