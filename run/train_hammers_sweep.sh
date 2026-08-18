#!/usr/bin/env bash
# Sequential HAMMERS training of all 13 model/dims combinations on GPU #1.
# Each run finishes before the next begins, preventing GPU OOM.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
set -o pipefail

mkdir -p runhammers
SUMMARY=runhammers/sweep_summary.log
echo "HAMMERS sweep started: $(date)" | tee -a "$SUMMARY"

MODELS=(
  cerebnet_hammers
  acapulco_hammers
  marin_hammers
  classical_cnn_hammers
  mipaim_unet_hammers
  swin_unetr_hammers
  qcnn_pennylane_hammers
  qcnn_qiskit_hammers
  hybrid_qcnn_hammers
  quaternion_unet_hammers
  quaternion_acapulco_hammers
  quaternion_marin_hammers
  quaternion_mipaim_hammers
)

for model_cfg in "${MODELS[@]}"; do
  model_name="${model_cfg/_hammers/}"
  for dims in 2 3; do
    if [ "$dims" = "3" ]; then
      extra="training.batch_size=1"
    else
      extra=""
    fi
    log="runhammers/${model_name}_${dims}d.log"
    echo "=========================================="
    echo "START: ${model_name} ${dims}D  ->  ${log}  $(date +%H:%M:%S)"
    echo "=========================================="
    if python run/pipeline/train_hammers.py \
        +experiment=exp_hammers \
        model=${model_cfg} \
        dataset.spatial_dims=${dims} \
        ${extra} \
        hydra.run.dir="$(pwd)/runhammers/${model_name}_${dims}d" \
        2>&1 | tee "${log}"; then
      status="OK  "
    else
      status="FAIL"
    fi
    echo "${status}  ${model_name} ${dims}D  $(date +%H:%M:%S)" | tee -a "$SUMMARY"
  done
done

echo "HAMMERS sweep finished: $(date)" | tee -a "$SUMMARY"
echo "All 26 runs attempted."