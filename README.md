# Quantum CNN vs Classical CNN: Benchmarking Study

Systematic comparison of quantum convolutional neural networks (QCNNs) against classical CNNs, investigating parameter efficiency, noise robustness, scaling behavior, and encoding strategies.

## Research Questions

- Do QCNNs outperform classical CNNs in the low-parameter regime?
- Does quantum entanglement provide noise robustness advantages?
- How does quantum advantage scale with number of output classes?
- Which data encoding strategy best leverages quantum Hilbert space?

## Project Structure

```
quant-cnn/
├── conf/                       # Hydra configs
│   ├── config.yaml             # Base config
│   ├── dataset/                # MNIST, Fashion-MNIST, CIFAR-10
│   ├── model/                  # Classical, Quantum, Hybrid models
│   └── experiment/             # 7 experiment configs
├── src/
│   ├── data_module/            # Dataset loading, transforms, noise
│   ├── model_module/
│   │   ├── classical/          # Standard CNN baselines
│   │   ├── quantum/            # PennyLane & Qiskit QCNNs
│   │   └── hybrid/             # Classical-Quantum-Classical hybrid
│   ├── trainer_module/         # Training loop with early stopping
│   └── utils/                  # Metrics, visualization, reproducibility
├── run/
│   ├── pipeline/               # train.py, compare.py
│   └── outputs/                # Experiment outputs (gitignored)
└── plan/                       # Experiment design documents
```

## Models

| Model | Type | Framework | Description |
|-------|------|-----------|-------------|
| `classical_cnn` | Classical | PyTorch | Standard CNN (Conv-BN-ReLU-Pool + FC) |
| `classical_cnn_small` | Classical | PyTorch | Parameter-matched small CNN |
| `qcnn_pennylane` | Quantum | PennyLane | Quantum circuit with classical pre-processing |
| `qcnn_qiskit` | Quantum | Qiskit | Same architecture via Qiskit ML |
| `hybrid_qcnn` | Hybrid | PennyLane | Classical Encoder → Quantum Layer → Classical Decoder |

## Experiments

| # | Name | Goal |
|---|------|------|
| 1 | Baseline | Binary classification baseline comparison |
| 2 | Parameter Efficiency | Fair comparison at matched parameter budgets |
| 3 | Multi-class Scaling | Quantum advantage vs number of classes |
| 4 | Noise Robustness | Noise resilience comparison |
| 5 | Qubit Scaling | Optimal qubits/depth, barren plateau detection |
| 6 | Encoding Study | Amplitude vs Angle vs IQP encoding |
| 7 | Framework Comparison | PennyLane vs Qiskit runtime comparison |

See [`plan/experiment_plan.md`](plan/experiment_plan.md) for full details.

## Quick Start

```bash
# Install dependencies
uv sync

# Run baseline experiment
python run/pipeline/train.py experiment=exp1_baseline model=classical_cnn
python run/pipeline/train.py experiment=exp1_baseline model=qcnn_pennylane

# Hydra multirun sweep
python run/pipeline/train.py -m experiment=exp5_qubit_scaling \
    model.quantum.num_qubits=4,6,8,10,12 model.quantum.num_layers=2,4,6,8

# Compare results
python run/pipeline/compare.py
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.2
- PennyLane >= 0.42
- Qiskit >= 1.0
