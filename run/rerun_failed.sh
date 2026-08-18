#!/usr/bin/env bash
set -euo pipefail
cd /home/ai/claude_research/qcnn-vs-cnn-medical-imaging
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

SUMMARY="run/outputs/rerun_failed_summary.log"
echo "Rerun of failed models started: $(date)" | tee "$SUMMARY"

MODELS=(
  "qcnn_pennylane_brainstem:2:training.batch_size=16"
  "qcnn_pennylane_brainstem:3:training.batch_size=2"
  "hybrid_qcnn_brainstem:2:training.batch_size=16"
  "hybrid_qcnn_brainstem:3:training.batch_size=2"
)

for entry in "${MODELS[@]}"; do
  IFS=':' read -r model_cfg dims extra <<< "$entry"
  model_name="${model_cfg/_brainstem/}"
  ckpt_dir="run/checkpoints/${model_name}_${dims}d"
  tb_dir="run/tb_logs/${model_name}_${dims}d"
  log="run/outputs/${model_name}_${dims}d_rerun.log"
  mkdir -p "$ckpt_dir" "$tb_dir"

  echo ""
  echo "=============================================="
  echo "[${model_name}] ${dims}D"
  echo "    log:        $log"
  echo "    started:    $(date +%H:%M:%S)"
  echo "=============================================="

  if python run/pipeline/train_medical.py       +experiment=exp_brainstem       model="${model_cfg}"       dataset.spatial_dims="${dims}"       logging.tensorboard=true       logging.tensorboard_dir="${tb_dir}"       ${extra}       2>&1 | tee "$log"; then
    status="OK  "
  else
    status="FAIL"
  fi

  src_dir="run/outputs/${model_name}_${dims}d"
  if [ -f "${src_dir}/best_model.pth" ]; then
      cp "${src_dir}/best_model.pth"  "${ckpt_dir}/best_model.pth"
      cp "${src_dir}/final_model.pth" "${ckpt_dir}/final_model.pth"
      cp "${src_dir}/results.json"    "${ckpt_dir}/results.json"
      echo "    checkpoints copied to ${ckpt_dir}"
  else
      echo "    WARNING: best_model.pth not found in ${src_dir}"
  fi

  echo "${status}  ${model_name} ${dims}D  $(date +%H:%M:%S)" | tee -a "$SUMMARY"
done

echo ""
echo "All reruns finished at $(date)" | tee -a "$SUMMARY"
