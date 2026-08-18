#!/usr/bin/env bash
# Full HAMMERS sweep rerun -- single worker on GPU 1 (GPU 0 reserved for ComfyUI).
# Runs all 26 configs (13 models x 2 dims) from scratch, using the bbox crop (164x134x104).
# Launch inside tmux:  bash run/train_hammers_sweep3.sh
set -o pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNDIR=runhammers
SUMMARY="$RUNDIR/sweep3_summary.log"
mkdir -p "$RUNDIR"
echo "=== Sweep3 started: $(date) ===" | tee -a "$SUMMARY"

run_one() {  # model_cfg dims
  local cfg="$1" dims="$2"
  local model_name="${cfg/_hammers/}"
  local log="$RUNDIR/${model_name}_${dims}d.sweep3.log"
  echo "START|${model_name}|${dims}D|$(date +%F_%H:%M:%S)" | tee -a "$SUMMARY"
  if CUDA_VISIBLE_DEVICES=1 python run/pipeline/train_hammers.py       +experiment=exp_hammers       model="${cfg}"       dataset.spatial_dims="${dims}"       $( [ "$dims" = "3" ] && echo "training.batch_size=1" )       hydra.run.dir="$(pwd)/$RUNDIR/${model_name}_${dims}d"       2>&1 | tee "$log"; then
    echo "OK  |${model_name}|${dims}D|$(date +%F_%H:%M:%S)" | tee -a "$SUMMARY"
  else
    echo "FAIL|${model_name}|${dims}D|$(date +%F_%H:%M:%S)" | tee -a "$SUMMARY"
  fi
}

# All 13 models x 2 dims. 3D first (fast, headline results), quantum 2D last (slowest).
for dims in 3 2; do
  for cfg in acapulco_hammers cerebnet_hammers classical_cnn_hammers marin_hammers              mipaim_unet_hammers swin_unetr_hammers quaternion_unet_hammers              quaternion_acapulco_hammers quaternion_marin_hammers quaternion_mipaim_hammers              qcnn_qiskit_hammers qcnn_pennylane_hammers hybrid_qcnn_hammers; do
    run_one "$cfg" "$dims"
  done
done
echo "=== Sweep3 finished: $(date) ===" | tee -a "$SUMMARY"
