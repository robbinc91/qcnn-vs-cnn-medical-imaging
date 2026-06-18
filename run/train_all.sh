#!/usr/bin/env bash
# Sequential training of all 18 model/dims combinations.
# Each run finishes before the next begins, preventing GPU OOM.
# Failures in one run are logged but do NOT abort the remaining runs.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
set -o pipefail

mkdir -p run/outputs
SUMMARY=run/outputs/sweep_summary.log
echo "Sweep started: $(date)" | tee -a "$SUMMARY"

MODELS=(
  cerebnet_brainstem
  acapulco_brainstem
  marin_brainstem
  classical_cnn_brainstem
  mipaim_unet_brainstem
  swin_unetr_brainstem
  qcnn_pennylane_brainstem
  qcnn_qiskit_brainstem
  hybrid_qcnn_brainstem
)

for model_cfg in "${MODELS[@]}"; do
  model_name="${model_cfg/_brainstem/}"
  for dims in 2 3; do
    # 3-D volumes are memory-heavy: drop batch size to avoid GPU OOM.
    if [ "$dims" = "3" ]; then
      extra="training.batch_size=2"
    else
      extra=""
    fi
    log="run/outputs/${model_name}_${dims}d.log"
    echo "=========================================="
    echo "START: ${model_name} ${dims}D  ->  ${log}  $(date +%H:%M:%S)"
    echo "=========================================="
    if python run/pipeline/train_medical.py \
        +experiment=exp_brainstem \
        model=${model_cfg} \
        dataset.spatial_dims=${dims} \
        ${extra} \
        2>&1 | tee "${log}"; then
      status="OK  "
    else
      status="FAIL"
    fi
    echo "${status}  ${model_name} ${dims}D  $(date +%H:%M:%S)" | tee -a "$SUMMARY"
  done
done

echo "Sweep finished: $(date)" | tee -a "$SUMMARY"
echo "All 18 runs attempted."
