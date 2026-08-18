#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="$PWD"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p "$BASE/run/checkpoints" "$BASE/run/outputs" "$BASE/run/tb_logs"
SUMMARY="$BASE/run/outputs/sweep_summary.log"
echo "Sweep started: $(date)" | tee -a "$SUMMARY"
echo "GPU: $($BASE/.venv/bin/python3 -c 'import torch; print(torch.cuda.get_device_name(0))')" | tee -a "$SUMMARY"

MODELS=(
  acapulco_brainstem
  cerebnet_brainstem
  classical_cnn_brainstem
  marin_brainstem
  mipaim_unet_brainstem
  swin_unetr_brainstem
  quaternion_unet_brainstem
  quaternion_acapulco_brainstem
  quaternion_marin_brainstem
  quaternion_mipaim_brainstem
  qcnn_pennylane_brainstem
  qcnn_qiskit_brainstem
  hybrid_qcnn_brainstem
)

TOTAL=${#MODELS[@]}
RUN=0
for model_cfg in "${MODELS[@]}"; do
  model_name="${model_cfg/_brainstem/}"
  for dims in 2 3; do
    RUN=$((RUN + 1))
    ckpt_dir="$BASE/run/checkpoints/${model_name}_${dims}d"
    tb_dir="$BASE/run/tb_logs/${model_name}_${dims}d"
    mkdir -p "$ckpt_dir" "$tb_dir"

    if [ "$dims" = "3" ]; then
      extra="training.batch_size=2"
    else
      extra=""
    fi
    log="$BASE/run/outputs/${model_name}_${dims}d.log"

    echo ""
    echo "=============================================="
    echo "[$RUN/$((TOTAL * 2))] ${model_name} ${dims}D"
    echo "    log:        $log"
    echo "    checkpoints: $ckpt_dir"
    echo "    tensorboard: $tb_dir"
    echo "    started:    $(date +%H:%M:%S)"
    echo "=============================================="

    cd "$BASE"
    if python run/pipeline/train_medical.py \
        +experiment=exp_brainstem \
        model="${model_cfg}" \
        dataset.spatial_dims="${dims}" \
        logging.tensorboard=true \
        logging.tensorboard_dir="${tb_dir}" \
        ${extra} \
        2>&1 | tee "$log"; then
      status="OK  "
    else
      status="FAIL"
    fi

    # Copy checkpoints to run/checkpoints/<model>/
    src_dir="$BASE/run/outputs/${model_name}_${dims}d"
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
done

echo ""
echo "=============================================="
echo "All $((TOTAL * 2)) runs finished at $(date)"
echo "Running per-subject evaluation..."
echo "=============================================="

cd "$BASE"
python run/pipeline/eval_per_subject2.py 2>&1 | tee "$BASE/run/outputs/eval_per_subject.log"
python run/pipeline/eval_fullvol_2d.py 2>&1 | tee "$BASE/run/outputs/eval_fullvol_2d.log"

echo "All done at $(date)" | tee -a "$SUMMARY"
echo "$((TOTAL * 2)) runs completed." | tee -a "$SUMMARY"
