# Quantum CNN vs Classical CNN for Medical Image Segmentation

Systematic benchmarking of **13 segmentation models** — classical CNNs, quaternion CNNs, quantum QCNNs (PennyLane and Qiskit), and a hybrid quantum–classical architecture — on two brain-MRI segmentation tasks: **brainstem** (3 structures) and **HAMMERS** (8 parietal-lobe regions). Every model is trained in both **2D axial-slice** and **full 3D volumetric** mode with per-subject Dice evaluation and statistical testing (Friedman omnibus + bootstrap-paired comparisons).

## Highlights

- **3D consistently beats 2D** in every single configuration on both datasets.
- **Classical and quaternion models are the strongest performers.** On HAMMERS, `marin_3d` is best 3D (DSC 0.804), `quaternion_marin_2d` is best 2D (DSC 0.790); on brainstem, `acapulco_3d` (0.933) and `cerebnet_2d` (0.917) lead.
- **3D model differences are largely not statistically significant** (Friedman 3D: HAMMERS p=0.356, brainstem p=0.026).
- **Quantum models match classical performance in 3D** (brainstem pooled 3D Δ=-0.012, 95% CI [-0.036, +0.002]) but the **2D comparison is confounded**: the three 2D quantum/hybrid models only exist at ~6.1M parameters versus ~17M for every other model (see [Known limitations](#known-limitations)).
- All 104 final + best checkpoints (52 models × 2) are released as GitHub release assets, sha256-verified against the originals.

## Datasets

### Brainstem (private)
- 39 subjects (a31–a70, excluding a37), T1-weighted brainstem MRI volumes, 3 structures + background: **medulla, pons, mesencephalon** (4 classes).
- Input volumes `80×80×96`; 2D mode segments axial slices that contain ≥ 20 foreground voxels.
- Split: **27 / 6 / 6** (train / val / test), seed 42.
- Note: 2D evaluation excludes empty axial slices, so 2D and 3D Dice are not directly comparable.

### HAMMERS (public atlas)
- 30 subjects (a01–a30), **8 parietal-lobe regions** + background (9 classes), from the HAMMERS atlas.
  - Hammers A. et al. *"Three-dimensional maximum probability atlas of the human brain, with particular reference to the temporal lobe"* — Human Brain Mapping 19(4):224–247, 2003. DOI: [10.1002/hbm.10123](https://doi.org/10.1002/hbm.10123)
  - Gousias I. et al. *"Automatic segmentation of brain MRIs of 2-year-olds into 83 regions of interest"* — NeuroImage 40(2):672–684, 2008. DOI: [10.1016/j.neuroimage.2007.11.034](https://doi.org/10.1016/j.neuroimage.2007.11.034)
- Full volumes are `186×199×170`. To make 3D training feasible on a single RTX 3090, training crops each volume with a **per-subject label-bounding-box crop + 6-voxel margin → 164×134×104** (`_bbox_crop` in `src/data_module/hammers.py`). Verified to lose zero label voxels; peak GPU memory drops from OOM to 6.5 GiB (`marin_3d`) / 19.3 GiB (`mipaim_unet_3d`) at batch size 1.
- Split: 15% val / 15% test (seed 42); per-subject evaluation on the 5 test subjects.

> Data files are **not** distributed in this repository (medical-imaging policy, see `.gitignore`). Brainstem data is private; HAMMERS NIfTI volumes can be obtained from the public HAMMERS atlas (see citations above). Expected layout: `datatrain/aXX.nii.gz` + `datatrain/aXX-seg.nii.gz`, and `datatrain_hammers/aXX.nii.gz` + `datatrain_hammers/aXX-seg.nii.gz`.

## Models (13)

All models are parameter-matched at **~17M parameters** (encoder channels `[80, 160, 320]`, 3D adds a depth axis) *except* the three 2D quantum/hybrid models (see limitations).

| Model | Type | Framework | Notes |
|---|---|---|---|
| `classical_cnn` | Classical | PyTorch | Standard Conv-BN-ReLU encoder + decoder |
| `marin` | Classical | PyTorch | U-Net-style baseline |
| `acapulco` | Classical | PyTorch | U-Net-style baseline |
| `cerebnet` | Classical | PyTorch | Cerebellar/3D-inspired CNN |
| `mipaim_unet` | Classical | MONAI | MONAI U-Net |
| `swin_unetr` | Classical | MONAI | MONAI Swin UNETR (transformer) |
| `quaternion_unet` | Quaternion | PyTorch | Quaternion-valued U-Net |
| `quaternion_marin` | Quaternion | PyTorch | Quaternion analogue of marin |
| `quaternion_acapulco` | Quaternion | PyTorch | Quaternion analogue of acapulco |
| `quaternion_mipaim` | Quaternion | PyTorch | Quaternion analogue of mipaim U-Net |
| `qcnn_pennylane` | Quantum | PennyLane | 8-qubit QCNN layer + classical U-Net backbone |
| `qcnn_qiskit` | Quantum | Qiskit ML | Same architecture via Qiskit |
| `hybrid_qcnn` | Hybrid | PennyLane | Classical encoder → quantum layer → classical decoder |

Architectures live in `src/model_module/{classical,quaternion,quantum,hybrid}/`; quaternion math in `src/model_module/quaternion/_quaternion_ops.py`.

## Results

Per-subject Dice (mean ± std). Brainstem: 6 test subjects; HAMMERS: 5 test subjects. Full per-subject CSVs and statistics are in `run/outputs/analysis/` (brainstem) and `runhammers/analysis/` (HAMMERS).

### Brainstem

| 2D | DSC | | 3D | DSC |
|---|---|---|---|---|
| cerebnet_2d | **0.917** ± 0.012 | | acapulco_3d | **0.933** ± 0.014 |
| quaternion_marin_2d | 0.916 ± 0.015 | | marin_3d | 0.930 ± 0.014 |
| marin_2d | 0.909 ± 0.013 | | quaternion_acapulco_3d | 0.927 ± 0.017 |
| mipaim_unet_2d | 0.908 ± 0.017 | | classical_cnn_3d | 0.926 ± 0.013 |
| quaternion_acapulco_2d | 0.907 ± 0.013 | | quaternion_marin_3d | 0.926 ± 0.015 |
| swin_unetr_2d | 0.907 ± 0.015 | | mipaim_unet_3d | 0.923 ± 0.023 |
| qcnn_qiskit_2d (6M) | 0.906 ± 0.009 | | swin_unetr_3d | 0.920 ± 0.017 |
| hybrid_qcnn_2d (6M) | 0.905 ± 0.011 | | quaternion_mipaim_3d | 0.919 ± 0.014 |
| acapulco_2d | 0.905 ± 0.012 | | quaternion_unet_3d | 0.917 ± 0.025 |
| classical_cnn_2d | 0.904 ± 0.011 | | hybrid_qcnn_3d | 0.915 ± 0.040 |
| quaternion_mipaim_2d | 0.903 ± 0.021 | | cerebnet_3d | 0.911 ± 0.045 |
| qcnn_pennylane_2d (6M) | 0.902 ± 0.015 | | qcnn_pennylane_3d | 0.911 ± 0.041 |
| quaternion_unet_2d | 0.902 ± 0.019 | | qcnn_qiskit_3d | 0.910 ± 0.052 |

**Friedman omnibus**: 2D chi²=24.76, **p=0.003** (10 models) · 3D chi²=23.16, **p=0.026** (13 models)
**Pooled quantum/hybrid vs classical (3D)**: Δ = −0.012, 95% CI [−0.036, +0.002], p=0.438 — not significant.

### HAMMERS

| 2D | DSC | | 3D | DSC |
|---|---|---|---|---|
| quaternion_marin_2d | **0.790** ± 0.008 | | marin_3d | **0.804** ± 0.019 |
| marin_2d | 0.782 ± 0.015 | | quaternion_marin_3d | 0.802 ± 0.027 |
| cerebnet_2d | 0.780 ± 0.008 | | mipaim_unet_3d | 0.800 ± 0.012 |
| acapulco_2d | 0.778 ± 0.018 | | quaternion_mipaim_3d | 0.800 ± 0.017 |
| mipaim_unet_2d | 0.769 ± 0.013 | | acapulco_3d | 0.799 ± 0.018 |
| quaternion_acapulco_2d | 0.765 ± 0.030 | | quaternion_acapulco_3d | 0.799 ± 0.023 |
| hybrid_qcnn_2d (6M) | 0.753 ± 0.021 | | classical_cnn_3d | 0.797 ± 0.020 |
| qcnn_pennylane_2d (6M) | 0.749 ± 0.025 | | qcnn_pennylane_3d | 0.796 ± 0.013 |
| quaternion_mipaim_2d | 0.749 ± 0.021 | | swin_unetr_3d | 0.794 ± 0.019 |
| quaternion_unet_2d | 0.740 ± 0.059 | | qcnn_qiskit_3d | 0.790 ± 0.011 |
| swin_unetr_2d | 0.728 ± 0.038 | | quaternion_unet_3d | 0.790 ± 0.020 |
| classical_cnn_2d | 0.722 ± 0.068 | | cerebnet_3d | 0.783 ± 0.025 |
| qcnn_qiskit_2d (6M) | 0.657 ± 0.048 | | hybrid_qcnn_3d | 0.777 ± 0.014 |

**Friedman omnibus**: 2D chi²=40.77, **p=0.000** (13 models) · 3D chi²=13.19, **p=0.356** — 3D models statistically indistinguishable.

### Key findings

- 3D > 2D for every model and dataset (larger context, no slice-discarding).
- Quaternion models match their classical counterparts (e.g. quaternion_marin ≈ marin in 3D on both datasets) while halving real-parameter count.
- In 3D, quantum models are statistically indistinguishable from classical ones; the 6M-parameter 2D quantum variants lag behind (especially `qcnn_qiskit_2d`, DSC 0.657 on HAMMERS).

## Pre-trained Checkpoints

All **104 checkpoints** (13 models × 2 dims × 2 datasets × final+best) are attached to the GitHub release:

[**`final-checkpoints`**](https://github.com/robbinc91/qcnn-vs-cnn-medical-imaging/releases/tag/final-checkpoints) — 3.11 GB (final) + 3.33 GB (best)

Naming: `{brainstem|hammers}_{model}_{2d|3d}_{final|best}_model.pth`

```bash
# e.g. restore the best brainstem 3D marin weights
curl -L -o marin_3d_best.pth \
  https://github.com/robbinc91/qcnn-vs-cnn-medical-imaging/releases/download/final-checkpoints/brainstem_marin_3d_best_model.pth
```

Every asset was sha256-verified against the local originals before the local copies were removed. Caveats:

- **Brainstem `qcnn_qiskit` checkpoints predate the `q_weight → qnn.weight` refactor** — loading them into the current model requires a small state-dict key remap.
- **2D quantum/hybrid checkpoints are the 6.1M-parameter variants** (the 17M 2D variants were never trained — see limitations).

## Installation

```bash
uv sync                 # installs from uv.lock (Python 3.12, PyTorch 2.6.0+cu124, PennyLane 0.44.1, Qiskit 2.4.0, MONAI)
source .venv/bin/activate
```

The reference environment (`environment.json`) ran on: Python 3.12.3, torch 2.6.0+cu124, CUDA 12.4, 2× NVIDIA RTX 3090, PennyLane 0.44.1, Qiskit 2.4.0. QCNN training is slow — `qcnn_qiskit_2d` takes ≈19 min/epoch.

## Usage

### Train a single model (brainstem)

```bash
python run/pipeline/train_medical.py experiment=exp_brainstem model=marin_brainstem dataset.spatial_dims=2
python run/pipeline/train_medical.py experiment=exp_brainstem model=marin_brainstem dataset.spatial_dims=3
```

### Train a single model (HAMMERS)

```bash
python run/pipeline/train_hammers.py experiment=exp_hammers model=marin_hammers dataset.spatial_dims=3
```

### Full sweeps

```bash
bash run/train_brainstem_sweep.sh    # all 13 models × 2 dims, brainstem
bash run/train_hammers_sweep3.sh     # all 13 models × 2 dims, HAMMERS (per-subject bbox crop)
```

### Evaluation & statistics

```bash
python run/pipeline/eval_per_subject2.py            # brainstem per-subject Dice
python run/pipeline/stats_analysis.py               # brainstem stats report + figures
python run/pipeline/eval_per_subject_hammers.py     # HAMMERS per-subject Dice
python run/pipeline/stats_analysis_hammers.py       # HAMMERS stats report + figures
python run/pipeline/gen_segmentation_screenshots_hammers.py   # segmentation overlays
```

Hydra configs are in `conf/` — pick any `conf/model/*_brainstem.yaml` / `*_hammers.yaml` with the matching experiment. Outputs land in `run/outputs/<model>/` (brainstem) or `runhammers/<model>/` (HAMMERS).

## Project Structure

```
├── conf/                      # Hydra configs
│   ├── config.yaml            # Base config
│   ├── dataset/               # brainstem, hammers (+ legacy MNIST/CIFAR experiments)
│   ├── experiment/            # exp_brainstem, exp_hammers (+ legacy exp1–7)
│   └── model/                 # 13 models × {brainstem, hammers} (+ legacy)
├── src/
│   ├── data_module/           # brainstem.py, hammers.py (bbox crop), transforms, noise
│   ├── model_module/
│   │   ├── classical/         # cnn, marin, acapulco, cerebnet, mipaim_unet, swin_unetr
│   │   ├── quaternion/        # quaternion UNet/Marin/Acapulco/MiPAIM + quaternion ops
│   │   ├── quantum/           # pennylane_qcnn, qiskit_qcnn
│   │   └── hybrid/            # hybrid_qcnn
│   ├── trainer_module/        # training loop, early stopping, checkpointing
│   └── utils/                 # losses, metrics, reproducibility, visualization
├── run/
│   ├── pipeline/              # train_medical, train_hammers, eval, stats, screenshots
│   ├── checkpoints/           # (removed after upload — see release)
│   └── outputs/               # brainstem results + analysis (gitignored)
├── runhammers/                # HAMMERS results + analysis (gitignored weights)
└── plan/, docs/               # design notes (legacy experiment docs)
```

`paper/` and the manuscript are maintained privately and are gitignored.

## Known Limitations

- **2D quantum/hybrid models are parameter-mismatched**: they exist only as 6.1M-parameter variants; the planned 17M 2D reruns (`run/rerun_qcnn_2d_17M.sh`, encoder `[80,160,320,480]`) were never executed. Any 2D quantum-vs-classical comparison should be interpreted with this confound in mind.
- **2D vs 3D Dice are not directly comparable** (2D drops near-empty axial slices).
- **Brainstem qiskit checkpoints** need a state-dict key remap after the model refactor.
- **Small test sets** (n=5–6 subjects) limit statistical power; 3D differences are mostly not significant.

## License

MIT (see `pyproject.toml`). Datasets are subject to their own terms; HAMMERS data should be cited as in [Datasets](#datasets).
