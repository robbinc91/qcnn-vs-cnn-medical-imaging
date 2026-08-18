#!/usr/bin/env bash
# Dual-GPU HAMMERS rerun sweep -- two workers in parallel (GPU 0 / GPU 1).
# Only reruns: 4 OOM'd 3D runs, qcnn_qiskit_2d (interrupted; now 12x faster),
# and the 11 never-started runs. Successes are skipped.
# Run inside tmux:  bash run/train_hammers_sweep2.sh
set -o pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNDIR=runhammers
SUMMARY="$RUNDIR/sweep2_summary.log"
mkdir -p "$RUNDIR"
echo "=== Sweep2 started: $(date) ===" | tee -a "$SUMMARY"

run_one() {  # gpu model_cfg dims
  local gpu="$1" cfg="$2" dims="$3"
  local model_name="${cfg/_hammers/}"
  local log="$RUNDIR/${model_name}_${dims}d.sweep2.log"
  echo "START|GPU${gpu}|${model_name}|${dims}D|$(date +%F_%H:%M:%S)" | tee -a "$SUMMARY"
  if CUDA_VISIBLE_DEVICES="$gpu" python run/pipeline/train_hammers.py \
      +experiment=exp_hammers \
      model="${cfg}" \
      dataset.spatial_dims="${dims}" \
      $( [ "$dims" = "3" ] && echo "training.batch_size=1" ) \
      hydra.run.dir="$(pwd)/$RUNDIR/${model_name}_${dims}d" \
      2>&1 | tee "$log"; then
    echo "OK  |GPU${gpu}|${model_name}|${dims}D|$(date +%F_%H:%M:%S)" | tee -a "$SUMMARY"
  else
    echo "FAIL|GPU${gpu}|${model_name}|${dims}D|$(date +%F_%H:%M:%S)" | tee -a "$SUMMARY"
  fi
}

# GPU 0: long-pole quantum 2D first, then quaternion 2D
gpu0_queue() {
  run_one 0 qcnn_qiskit_hammers 2
  run_one 0 hybrid_qcnn_hammers 2
  run_one 0 quaternion_unet_hammers 2
  run_one 0 quaternion_acapulco_hammers 2
}

# GPU 1: 3D reruns + all never-started 3D, then leftover quaternion 2D
gpu1_queue() {
  run_one 1 cerebnet_hammers 3
  run_one 1 marin_hammers 3
  run_one 1 mipaim_unet_hammers 3
  run_one 1 swin_unetr_hammers 3
  run_one 1 qcnn_qiskit_hammers 3
  run_one 1 hybrid_qcnn_hammers 3
  run_one 1 quaternion_unet_hammers 3
  run_one 1 quaternion_acapulco_hammers 3
  run_one 1 quaternion_marin_hammers 3
  run_one 1 quaternion_mipaim_hammers 3
  run_one 1 quaternion_marin_hammers 2
  run_one 1 quaternion_mipaim_hammers 2
}

gpu0_queue &
gpu1_queue &
wait
echo "=== Sweep2 finished: $(date) ===" | tee -a "$SUMMARY"
