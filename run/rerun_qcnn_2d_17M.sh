#!/usr/bin/env bash
# Re-run the 2-D quantum/hybrid models at ~17M params (4-stage encoder
# [80,160,320,480]), making the 2-D comparison parameter-matched with the
# ~17M classical baselines. 3-D configs are untouched; only the three 2-D
# quantum/hybrid runs are repeated. Old 6M results are in run/archive_6M_2d/.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
set -o pipefail

SUMMARY=run/outputs/rerun_17M_summary.log
echo "Rerun (2D, ~17M, enc=[80,160,320,480]) started: $(date)" | tee -a "$SUMMARY"

run_one () {
  local cfg="$1" override="$2" name="$3"
  local log="run/outputs/${name}_2d.log"
  echo "=========================================="
  echo "START: ${name} 2D (17M)  ->  ${log}  $(date +%H:%M:%S)"
  echo "=========================================="
  if python run/pipeline/train_medical.py \
      +experiment=exp_brainstem \
      model="${cfg}" \
      dataset.spatial_dims=2 \
      "${override}" \
      2>&1 | tee "${log}"; then
    echo "OK    ${name} 2D (17M)  $(date +%H:%M:%S)" | tee -a "$SUMMARY"
  else
    echo "FAIL  ${name} 2D (17M)  $(date +%H:%M:%S)" | tee -a "$SUMMARY"
  fi
}

run_one qcnn_pennylane_brainstem "model.enc_channels=[80,160,320,480]"                    qcnn_pennylane
run_one qcnn_qiskit_brainstem    "model.enc_channels=[80,160,320,480]"                    qcnn_qiskit
run_one hybrid_qcnn_brainstem    "model.classical_encoder.conv_channels=[80,160,320,480]" hybrid_qcnn

echo "Rerun (2D, ~17M) finished: $(date)" | tee -a "$SUMMARY"
echo "All 3 reruns attempted."
