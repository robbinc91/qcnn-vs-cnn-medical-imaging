# Experiment Plan: Quantum CNN vs Classical CNN

## Research Question

**What improvements can quantum-based convolutional neural networks bring over classical CNNs, and under what conditions do these advantages manifest?**

## Hypotheses

1. **H1 (Parameter Efficiency)**: QCNNs achieve higher accuracy than classical CNNs when both operate under the same parameter budget, due to the exponential expressibility of parameterized quantum circuits.
2. **H2 (Noise Robustness)**: Quantum entanglement-based feature extraction provides greater robustness to input noise compared to classical convolutions.
3. **H3 (Scaling Limits)**: The quantum advantage diminishes as the number of output classes increases, due to measurement bottlenecks.
4. **H4 (Encoding Impact)**: Data encoding strategy is the dominant factor in QCNN performance, more so than circuit depth or entanglement topology.

---

## Experiment Matrix

### Exp 1: Baseline Binary Classification
| Factor | Value |
|--------|-------|
| **Goal** | Establish performance baseline |
| **Dataset** | MNIST (digits 3 vs 6), N=2000 |
| **Models** | `classical_cnn`, `qcnn_pennylane`, `hybrid_qcnn` |
| **Metrics** | Accuracy, F1, convergence speed, parameter count |
| **Config** | `conf/experiment/exp1_baseline.yaml` |
| **Expected** | QCNN competitive or better with fewer params |

### Exp 2: Parameter Efficiency
| Factor | Value |
|--------|-------|
| **Goal** | Test H1 — fair comparison at matched param budgets |
| **Dataset** | MNIST (3 vs 6), N=2000 |
| **Models** | `classical_cnn_small` (param-matched) vs `qcnn_pennylane` |
| **Metrics** | Accuracy at 50/100/500/1000/5000 parameters |
| **Config** | `conf/experiment/exp2_param_efficiency.yaml` |
| **Analysis** | Plot accuracy vs parameter count (log scale) |
| **Expected** | QCNN dominates in <500 param regime |

### Exp 3: Multi-class Scaling
| Factor | Value |
|--------|-------|
| **Goal** | Test H3 — how quantum advantage scales with classes |
| **Dataset** | MNIST subsets: 2-class, 4-class, 10-class |
| **Models** | `hybrid_qcnn` vs `classical_cnn` |
| **Metrics** | Accuracy, F1-macro, per-class precision |
| **Config** | `conf/experiment/exp3_multiclass.yaml` |
| **Sweep** | `dataset.binary_classes=[3,6]`, `[0,1,3,6]`, `null` |
| **Expected** | Quantum advantage strongest at 2-class, degrades at 10 |

### Exp 4: Noise Robustness
| Factor | Value |
|--------|-------|
| **Goal** | Test H2 — noise resilience of quantum features |
| **Dataset** | MNIST (3 vs 6) with injected noise |
| **Noise types** | Gaussian (σ=0-0.5), Salt & Pepper (p=0-0.2), Rotation (0-30°) |
| **Models** | `classical_cnn`, `hybrid_qcnn` |
| **Metrics** | Accuracy degradation curves |
| **Config** | `conf/experiment/exp4_noise_robustness.yaml` |
| **Analysis** | Plot accuracy vs noise level for each type |
| **Expected** | QCNN degrades more gracefully under Gaussian noise |

### Exp 5: Qubit & Circuit Depth Scaling
| Factor | Value |
|--------|-------|
| **Goal** | Find optimal circuit configuration, detect barren plateaus |
| **Dataset** | MNIST (3 vs 6), N=1000 |
| **Sweep** | qubits ∈ {4, 6, 8, 10, 12}, layers ∈ {2, 4, 6, 8} |
| **Metrics** | Accuracy, gradient variance (barren plateau indicator) |
| **Config** | `conf/experiment/exp5_qubit_scaling.yaml` |
| **Analysis** | Heatmap of accuracy over (qubits × layers) |
| **Expected** | Sweet spot around 8 qubits / 4 layers; gradient vanishing at 12+ qubits with 8+ layers |

### Exp 6: Data Encoding Strategy
| Factor | Value |
|--------|-------|
| **Goal** | Test H4 — impact of encoding method |
| **Dataset** | Fashion-MNIST (T-shirt vs Shirt — hard pair) |
| **Encodings** | Amplitude, Angle, IQP |
| **Models** | `qcnn_pennylane` with each encoding |
| **Metrics** | Accuracy, training stability |
| **Config** | `conf/experiment/exp6_encoding_study.yaml` |
| **Expected** | Amplitude encoding best for image data; angle encoding most parameter-efficient |

### Exp 7: Framework Comparison
| Factor | Value |
|--------|-------|
| **Goal** | Quantify PennyLane vs Qiskit differences |
| **Dataset** | MNIST (3 vs 6), N=1000 |
| **Models** | `qcnn_pennylane` vs `qcnn_qiskit` (matched architecture) |
| **Metrics** | Accuracy, wall-clock training time, memory usage |
| **Config** | `conf/experiment/exp7_framework_comparison.yaml` |
| **Expected** | PennyLane slightly higher accuracy, Qiskit faster on statevector sim |

---

## Execution Order

```
Phase 1 (Baselines):     Exp 1 → Exp 2
Phase 2 (Deep dives):    Exp 5, Exp 6 (parallel)
Phase 3 (Applications):  Exp 3, Exp 4 (parallel)
Phase 4 (Engineering):   Exp 7
```

## Run Commands

```bash
# Setup
uv sync

# Phase 1
python run/pipeline/train.py experiment=exp1_baseline model=classical_cnn
python run/pipeline/train.py experiment=exp1_baseline model=qcnn_pennylane
python run/pipeline/train.py experiment=exp1_baseline model=hybrid_qcnn

python run/pipeline/train.py experiment=exp2_param_efficiency model=classical_cnn_small
python run/pipeline/train.py experiment=exp2_param_efficiency model=qcnn_pennylane

# Phase 2 (Hydra multirun)
python run/pipeline/train.py -m experiment=exp5_qubit_scaling \
    model.quantum.num_qubits=4,6,8,10,12 model.quantum.num_layers=2,4,6,8

python run/pipeline/train.py -m experiment=exp6_encoding_study \
    model.encoding.method=amplitude,angle,iqp

# Phase 3
python run/pipeline/train.py experiment=exp3_multiclass
python run/pipeline/train.py -m experiment=exp4_noise_robustness \
    noise.gaussian_std=0.0,0.1,0.2,0.3,0.5

# Phase 4
python run/pipeline/train.py experiment=exp7_framework_comparison model=qcnn_pennylane
python run/pipeline/train.py experiment=exp7_framework_comparison model=qcnn_qiskit

# Compare all
python run/pipeline/compare.py
```

## Key References

1. Cong et al. "Quantum Convolutional Neural Networks" (Nature Physics, 2019)
2. Hybrid QCQ-CNN (Scientific Reports, 2025) — alternating quantum-classical layers
3. 49-qubit QCNN on IBM Heron (arXiv:2505.05957, May 2025) — 96% MNIST accuracy
4. Barren plateau mitigation survey (Springer, 2025)
5. PennyLane 0.42 vs Qiskit 1.0 benchmark comparison (2025)
