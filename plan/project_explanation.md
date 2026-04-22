# Quantum CNN vs Classical CNN: Full Project Explanation

## 1. What This Project Is About

This project systematically investigates whether **quantum convolutional neural networks (QCNNs)** offer measurable improvements over **classical CNNs** for image classification. Rather than accepting hype or dismissing quantum ML outright, we design controlled experiments that isolate specific variables — parameter count, noise, encoding strategy, circuit depth — to determine *when* and *why* quantum approaches help (or don't).

The core question: **Under what conditions does running part of a neural network on quantum hardware (or a quantum simulator) beat a purely classical network?**

---

## 2. Background: Why Quantum CNNs?

### Classical CNNs in 30 Seconds

A classical CNN slides learned filters (kernels) across an image, detecting local patterns like edges, textures, and shapes. Stacking convolutional layers builds up a hierarchy: edges → parts → objects. Pooling layers reduce spatial dimensions, and fully-connected layers at the end map features to class predictions. The whole thing is trained end-to-end via backpropagation.

### What Quantum Brings to the Table

A quantum computer operates on **qubits** instead of classical bits. A qubit can be in a **superposition** of 0 and 1 simultaneously, and multiple qubits can be **entangled** — meaning their states are correlated in ways impossible classically. This gives quantum circuits two theoretical advantages:

1. **Exponential state space**: N qubits span a 2^N dimensional Hilbert space. An 8-qubit circuit naturally operates in a 256-dimensional space using only 8 physical units. A classical network needs 256 neurons to match this.

2. **Entanglement as a feature extractor**: Entangling gates create correlations between qubits that have no direct classical analogue. These can capture complex feature interactions that would require many classical parameters to approximate.

A **Quantum CNN (QCNN)** replaces (or augments) the classical convolutional layers with **parameterized quantum circuits (PQCs)**. Classical image data is encoded into qubit states, processed through trainable quantum gates, and measured to produce classical outputs that feed into the rest of the pipeline.

### The Catch

Current quantum hardware is **noisy** (NISQ era — Noisy Intermediate-Scale Quantum). Qubits decohere, gates introduce errors, and we can only run shallow circuits before noise destroys the signal. Simulation on classical computers is exponentially expensive in the number of qubits. So practical QCNNs today operate with 4-16 qubits and rely on hybrid architectures that combine classical preprocessing with a small quantum bottleneck.

---

## 3. The Five Models We Built

### 3.1 Classical CNN (`classical_cnn`)

**File**: `src/model_module/classical/cnn.py`

Standard PyTorch CNN serving as our upper-bound baseline:

```
Input Image (28x28x1)
  → Conv2d(1→16, 3x3) → BatchNorm → ReLU → MaxPool(2x2)    [14x14x16]
  → Conv2d(16→32, 3x3) → BatchNorm → ReLU → MaxPool(2x2)   [7x7x32]
  → Flatten                                                    [1568]
  → Linear(1568→128) → ReLU → Dropout(0.25)                   [128]
  → Linear(128→num_classes)                                    [2 or 10]
```

This is a fairly standard small CNN. With batch normalization and two conv layers, it has around **~25K parameters** on MNIST. It should achieve >98% on binary MNIST easily — the point is to see how QCNNs compare.

### 3.2 Small Classical CNN (`classical_cnn_small`)

**File**: Same file, same class (config-driven)

A deliberately crippled CNN with the same architecture but fewer channels:

```
Conv2d(1→4) → ReLU → MaxPool → Conv2d(4→8) → ReLU → MaxPool → FC(8*7*7→32) → FC(32→2)
```

This produces roughly **~2K parameters** — matching the parameter count of the QCNN. The purpose: if the QCNN beats this but not the full CNN, the advantage is purely about parameter efficiency, not about quantum magic.

### 3.3 PennyLane QCNN (`qcnn_pennylane`)

**File**: `src/model_module/quantum/pennylane_qcnn.py`

Our primary quantum model, built with the PennyLane framework:

```
Input Image (28x28)
  → Classical Preprocessor: Conv2d(1→4) → ReLU → AdaptiveAvgPool(4x4)   [64]
  → Linear Projection: FC(64→8) → tanh → scale to [-pi, pi]             [8 values]
  → Quantum Circuit (8 qubits, 4 layers):
      AngleEmbedding(8 values → 8 qubits)
      StronglyEntanglingLayers(weights)  ← these are the trainable params
      Measure: <Z_0>, <Z_1>, ..., <Z_7>                                  [8 values]
  → Linear Head: FC(8→num_classes)                                        [2 or 10]
```

**How the quantum circuit works step by step:**

1. **Angle Embedding**: Each of the 8 input values is encoded as a rotation angle on a qubit. Value 0.5 → rotate qubit by 0.5 radians. This maps classical data into quantum state space.

2. **StronglyEntanglingLayers**: This is the trainable part. Each layer applies:
   - Three rotation gates (Rot) on every qubit with learnable angles
   - CNOT gates connecting qubits in a pattern (linear, circular, or all-to-all)
   
   With 4 layers and 8 qubits, this creates 4 × 8 × 3 = **96 trainable quantum parameters**. Despite the small count, these parameters control rotations in a 256-dimensional Hilbert space.

3. **Measurement**: We measure the expectation value of the Pauli-Z operator on each qubit. This collapses the quantum state to 8 classical floating-point numbers in [-1, 1].

4. **Classical Head**: A simple linear layer maps 8 measurements to class logits.

The classical preprocessor is necessary because we can't encode a full 784-pixel image into 8 qubits directly. It acts as a learned dimensionality reducer — the Conv2d + AvgPool compresses 28x28 to 4x4x4=64 features, then a linear layer projects to 8.

**Total parameters**: ~400 classical (preprocessor + projection + head) + 96 quantum = **~500 total**.

### 3.4 Qiskit QCNN (`qcnn_qiskit`)

**File**: `src/model_module/quantum/qiskit_qcnn.py`

Same logical architecture as the PennyLane version, reimplemented in IBM's Qiskit framework. The key differences are:

- Uses **ZZFeatureMap** for encoding (creates entanglement during encoding itself)
- Uses **RealAmplitudes** ansatz (Ry rotations + CNOT entanglement)
- Connects to PyTorch via `TorchConnector` from `qiskit-machine-learning`
- Can optionally run on **real IBM quantum hardware** or noisy simulators

We include this to test whether the framework itself affects results — same math should give same answers, but implementation differences in gradient computation, numerical precision, and circuit optimization can matter.

### 3.5 Hybrid QCNN (`hybrid_qcnn`)

**File**: `src/model_module/hybrid/hybrid_qcnn.py`

The most sophisticated architecture, following the **QCQ (Quantum-Classical-Quantum)** paradigm from recent literature:

```
Classical Encoder                    Quantum Processor              Classical Decoder
─────────────────                    ─────────────────              ─────────────────
Conv2d(1→8) → ReLU → MaxPool        AngleEmbedding(8 qubits)       FC(8→64) → ReLU
Conv2d(8→16) → ReLU → MaxPool   →   StronglyEntanglingLayers   →   Dropout(0.2)
Flatten → FC(→8) → tanh*pi          Measure Z on all qubits        FC(64→num_classes)
```

The philosophy: let classical layers handle what they're good at (spatial feature extraction, dimensionality reduction) and let the quantum layer handle what it might be good at (finding complex feature correlations in compressed space). The classical decoder is deeper than in the pure QCNN, allowing richer post-processing of quantum measurements.

---

## 4. How Data Flows Through the System

### Configuration System (Hydra)

Everything is driven by YAML configs that compose together:

```
conf/
├── config.yaml              # Base: training params, evaluation, logging
├── dataset/mnist.yaml       # Which dataset, how many samples, which classes
├── model/qcnn_pennylane.yaml # Architecture: qubits, layers, encoding, preprocessor
└── experiment/exp1.yaml     # Override bundle: ties a dataset + model + specific params
```

When you run `python run/pipeline/train.py experiment=exp1_baseline model=qcnn_pennylane`, Hydra merges all these configs into a single `cfg` object. Every component reads from `cfg` — the dataset loader, model constructor, trainer, and evaluation.

### Training Pipeline (`run/pipeline/train.py`)

```
1. set_seed(42)                    # Reproducibility
2. log_environment()               # Record Python/PyTorch/PennyLane versions
3. DatasetFactory(cfg)             # → train_loader, val_loader, test_loader
4. ModelFactory(cfg)               # → model (classical or quantum)
5. Trainer(model, loaders, cfg)    # Configure optimizer, scheduler, metrics
6. trainer.train()                 # Training loop with early stopping
7. Save results.json + plots       # For later comparison
```

### Factory/Registry Pattern

Models self-register via decorators:

```python
@register_model("qcnn_pennylane")
class PennyLaneQCNN(nn.Module):
    ...
```

When `ModelFactory(cfg)` is called, it looks up `cfg.model.name` in the registry and instantiates the right class. This means adding a new model is just: write the class, decorate it, add a config YAML. No need to touch the training pipeline.

---

## 5. The Seven Experiments

### Experiment 1: Baseline

**Question**: How do the three model families compare on the simplest possible task?

We use MNIST digits 3 vs 6 (binary classification, 2000 samples). This is deliberately easy — both models should do well. What we measure is *how* they get there: convergence speed, final accuracy, and parameter count.

**What to look for**: If the QCNN reaches 95%+ accuracy with 500 parameters while the full classical CNN needs 25K, that's already an interesting finding about expressibility.

### Experiment 2: Parameter Efficiency

**Question**: When both models have the *same number* of trainable parameters, which wins?

This is the most important experiment. We match the classical CNN's parameter count to the QCNN's (~500) and compare accuracy. The hypothesis is that quantum parameters are more "expressive" per parameter because they operate in exponentially large Hilbert space.

**What to look for**: A crossover point — below X parameters, QCNN wins; above X, classical catches up. The literature suggests this crossover is around 500-1000 parameters.

### Experiment 3: Multi-class Scaling

**Question**: Does quantum advantage survive when moving from 2 classes to 10?

We run the hybrid QCNN on 2-class, 4-class, and 10-class MNIST. With 8 qubits and PauliZ measurements, we get 8 output values — mapping these to 10 classes is a tighter bottleneck than mapping to 2.

**What to look for**: Accuracy gap between quantum and classical should narrow with more classes. If the gap inverts (classical wins at 10 classes), it reveals a fundamental measurement bottleneck.

### Experiment 4: Noise Robustness

**Question**: Do quantum features degrade more gracefully under noisy inputs?

We train both models on clean data, then test on increasingly noisy versions. Three noise types:
- **Gaussian**: Random pixel perturbations (simulates sensor noise)
- **Salt & Pepper**: Random black/white pixels (simulates transmission errors)
- **Rotation**: Small random rotations (simulates alignment errors)

**What to look for**: Accuracy degradation curves. If the QCNN curve is flatter (degrades less), entanglement-based features may be inherently more robust. This would be a practical advantage for real-world deployment.

### Experiment 5: Qubit & Circuit Depth Scaling

**Question**: What's the optimal circuit configuration, and when do barren plateaus appear?

We sweep a 5x4 grid: qubits in {4, 6, 8, 10, 12} x layers in {2, 4, 6, 8}. For each combination, we train and measure both accuracy and **gradient variance** (the key barren plateau indicator — when gradients become exponentially small, the circuit is untrainable).

**What to look for**: A heatmap showing a "sweet spot" of high accuracy. Too few qubits/layers → underfitting. Too many → barren plateaus. The literature predicts trouble around 12+ qubits with deep circuits.

### Experiment 6: Data Encoding Strategy

**Question**: Does how we encode classical data into qubits matter more than the circuit itself?

Three encoding methods:
- **Angle Encoding**: Each feature → rotation angle on one qubit. Simple, 1:1 mapping.
- **Amplitude Encoding**: Encodes 2^N features into N qubits via state amplitudes. Dense but requires normalized inputs.
- **IQP Encoding**: Uses diagonal gates with feature products, creating non-linear encoding.

We test on Fashion-MNIST (T-shirt vs Shirt) — a harder pair than MNIST digits — to stress-test the encodings.

**What to look for**: If encoding choice changes accuracy by >5%, it confirms H4 (encoding dominates). This has direct practical implications for how to design QCNN pipelines.

### Experiment 7: Framework Comparison

**Question**: PennyLane vs Qiskit — does the implementation matter?

Same logical circuit, same dataset, same training procedure. We compare:
- Final test accuracy
- Wall-clock training time per epoch
- Peak memory usage

**What to look for**: They should be close on accuracy. PennyLane uses analytical gradients (adjoint differentiation) while Qiskit uses parameter-shift rules by default — this affects training speed. PennyLane integrates more natively with PyTorch, which may give it an edge.

---

## 6. How the Quantum Circuit Actually Works (Detailed)

This section explains the quantum mechanics at the code level for readers unfamiliar with quantum computing.

### Qubits and State

A single qubit is described by a state vector:

```
|psi> = alpha|0> + beta|1>
```

where alpha and beta are complex numbers with |alpha|^2 + |beta|^2 = 1. Measuring the qubit gives 0 with probability |alpha|^2 and 1 with probability |beta|^2.

With N=8 qubits, the combined state lives in a **256-dimensional** complex vector space (2^8 = 256). This is the source of quantum's theoretical advantage: 8 qubits can represent information that would require 256 classical neurons.

### What Our Circuit Does

Looking at the PennyLane QCNN's quantum circuit:

```python
# Step 1: Initialize qubits to |000...0>
# (happens automatically)

# Step 2: Encode data
qml.AngleEmbedding(inputs, wires=range(8))
# For each qubit i, applies RX(inputs[i])
# This rotates qubit i by an angle proportional to the input feature

# Step 3: Trainable layers (repeated 4 times)
qml.StronglyEntanglingLayers(weights, wires=range(8))
# Each layer does:
#   a) Rot(phi, theta, omega) on each qubit — 3 learnable angles per qubit
#   b) CNOT between qubit pairs — creates entanglement
# Entanglement means: qubit 0's state now depends on qubit 1's state
# This is how the circuit learns feature correlations

# Step 4: Measure
[qml.expval(qml.PauliZ(i)) for i in range(8)]
# For each qubit, measure the expectation of sigma_z
# Returns a value in [-1, 1]
# +1 = qubit mostly in |0> state
# -1 = qubit mostly in |1> state
```

### Why Gradients Work

The circuit is **differentiable** with respect to the weight angles. PennyLane computes gradients using either:

- **Parameter-shift rule**: Evaluate the circuit at weight+pi/2 and weight-pi/2, take the difference. Exact but requires 2 circuit evaluations per parameter.
- **Adjoint differentiation**: Reverse-mode autodiff on the unitary matrices. Faster for simulators.

This lets us train the quantum circuit with standard PyTorch optimizers (Adam, SGD) — the quantum weights are just `nn.Parameter` tensors.

### The Bottleneck: Sequential Processing

Look at the forward pass:

```python
q_out = torch.stack([
    torch.stack(self.q_layer.circuit(x[i], self.q_weights))
    for i in range(batch_size)
])
```

Each sample in the batch is processed **sequentially** through the quantum circuit. There's no batch-level parallelism because each circuit evaluation simulates a separate quantum system. This is the major practical limitation — a batch of 32 requires 32 sequential circuit evaluations, which is far slower than a classical matrix multiplication that processes the whole batch at once.

On real quantum hardware, you'd need to run 32 separate quantum programs. On simulators, each evaluation involves 256x256 matrix operations (for 8 qubits). This is why we subsample datasets to 1000-2000 examples.

---

## 7. Infrastructure Details

### Reproducibility

Every experiment records:
- **Random seed** (42 by default) — set for Python, NumPy, PyTorch, CUDA
- **Environment snapshot** — Python/PyTorch/PennyLane/Qiskit versions, GPU model
- **Resolved config** — the full merged Hydra config, saved as YAML
- **Training history** — loss/accuracy per epoch, stored in JSON

To reproduce any result: load the saved config and run with the same seed.

### Metrics Tracking

The `MetricsTracker` class monitors validation accuracy and implements early stopping. If accuracy doesn't improve for `patience` epochs (default 10), training stops. The best model weights are restored before test evaluation.

Tracked metrics: accuracy, F1-macro, precision, recall, confusion matrix.

### Visualization

Three plot types are generated automatically:
- **Training curves**: Loss and accuracy over epochs (train vs val)
- **Confusion matrix**: Heatmap of predictions vs true labels
- **Comparison plots**: Bar charts and scatter plots across experiments

The `compare.py` script scans all `run/outputs/` subdirectories, collects `results.json` files, and produces:
- A summary CSV table
- Accuracy comparison bar chart
- Parameter efficiency scatter plot (log-scale params vs accuracy)

### Noise Augmentation

The `NoiseAugmentation` class (`src/data_module/noise.py`) is designed for Experiment 4. It applies noise *after* normalization, directly to tensors. Three types:
- **Gaussian**: Adds N(0, sigma) noise, clips to [-1, 1]
- **Salt & Pepper**: Randomly sets pixels to min/max values
- **Rotation**: Applies random rotation within a degree range

Noise levels are swept via Hydra multirun, so each noise intensity gets its own output directory.

---

## 8. Expected Results and What They Mean

Based on the literature (Cong 2019, QCQ-CNN 2025, IBM Heron 2025):

| Experiment | Expected Outcome | Implication |
|-----------|------------------|-------------|
| Baseline | QCNN ~94%, CNN ~99% (QCNN fewer params) | Quantum is competitive but raw accuracy trails |
| Param Efficiency | QCNN wins below ~500 params | Quantum parameters are more expressive |
| Multi-class | Quantum advantage shrinks from 2→10 classes | Measurement bottleneck is real |
| Noise Robustness | QCNN 2-5% more robust to Gaussian noise | Entanglement captures robust features |
| Qubit Scaling | Best at 8 qubits / 4 layers | Barren plateaus limit scaling |
| Encoding | Amplitude > Angle > IQP | Encoding choice matters more than depth |
| Framework | PennyLane +1-2% accuracy, Qiskit 1.5x faster | Different optimization backends matter |

### The Honest Assessment

Quantum CNNs in 2025 are **not** going to replace classical CNNs for production image classification. The advantages are narrow:

- **Parameter efficiency** in the extreme low-parameter regime (useful for edge devices?)
- **Potential noise robustness** (needs more evidence)
- **Theoretical expressibility** that current hardware can't fully exploit

The real value of this research is understanding *where the boundary is* — at what problem size and parameter budget does quantum start helping, and at what point does classical scaling win? This informs when it will be worth revisiting quantum ML as hardware improves.

---

## 9. How to Run Everything

### Setup

```bash
cd quant-cnn
uv sync                    # Install all dependencies
```

### Run a Single Experiment

```bash
# Classical baseline
python run/pipeline/train.py experiment=exp1_baseline model=classical_cnn

# Quantum model
python run/pipeline/train.py experiment=exp1_baseline model=qcnn_pennylane

# Override any config from CLI
python run/pipeline/train.py experiment=exp1_baseline model=qcnn_pennylane \
    training.epochs=100 model.quantum.num_qubits=12
```

### Run a Sweep (Hydra Multirun)

```bash
# Sweep over qubit counts and circuit depths (20 combinations)
python run/pipeline/train.py -m experiment=exp5_qubit_scaling \
    model.quantum.num_qubits=4,6,8,10,12 \
    model.quantum.num_layers=2,4,6,8
```

### Run All Phases

```bash
# Phase 1: Baselines
python run/pipeline/train.py experiment=exp1_baseline model=classical_cnn
python run/pipeline/train.py experiment=exp1_baseline model=qcnn_pennylane
python run/pipeline/train.py experiment=exp1_baseline model=hybrid_qcnn
python run/pipeline/train.py experiment=exp2_param_efficiency model=classical_cnn_small
python run/pipeline/train.py experiment=exp2_param_efficiency model=qcnn_pennylane

# Phase 2: Circuit analysis (can run in parallel)
python run/pipeline/train.py -m experiment=exp5_qubit_scaling \
    model.quantum.num_qubits=4,6,8,10,12 model.quantum.num_layers=2,4,6,8
python run/pipeline/train.py -m experiment=exp6_encoding_study \
    model.encoding.method=amplitude,angle,iqp

# Phase 3: Applications (can run in parallel)
python run/pipeline/train.py experiment=exp3_multiclass
python run/pipeline/train.py -m experiment=exp4_noise_robustness \
    noise.gaussian_std=0.0,0.1,0.2,0.3,0.5

# Phase 4: Framework comparison
python run/pipeline/train.py experiment=exp7_framework_comparison model=qcnn_pennylane
python run/pipeline/train.py experiment=exp7_framework_comparison model=qcnn_qiskit

# Compare everything
python run/pipeline/compare.py
```

### Outputs

Each run creates a timestamped directory under `run/outputs/`:

```
run/outputs/2026-04-13/14-30-00/
├── .hydra/
│   ├── config.yaml           # Merged config
│   ├── hydra.yaml            # Hydra settings
│   └── overrides.yaml        # CLI overrides
├── environment.json          # Python/library versions
├── config_resolved.yaml      # Fully resolved config
├── results.json              # Metrics, history, params count
├── training_curves.png       # Loss/accuracy plots
└── main.log                  # Full training log
```

---

## 10. File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `src/model_module/classical/cnn.py` | ~95 | Classical CNN baseline (configurable channels, BN, dropout) |
| `src/model_module/quantum/pennylane_qcnn.py` | ~175 | PennyLane QCNN with classical preprocessor |
| `src/model_module/quantum/qiskit_qcnn.py` | ~135 | Qiskit QCNN with TorchConnector |
| `src/model_module/hybrid/hybrid_qcnn.py` | ~115 | Hybrid classical-quantum-classical model |
| `src/model_module/registry.py` | ~35 | Factory pattern for model instantiation |
| `src/data_module/dataset.py` | ~105 | Dataset loading, filtering, subsampling |
| `src/data_module/noise.py` | ~55 | Noise augmentation for robustness experiments |
| `src/data_module/transforms.py` | ~20 | Image preprocessing transforms |
| `src/trainer_module/trainer.py` | ~175 | Training loop, early stopping, checkpointing |
| `src/utils/metrics.py` | ~95 | Evaluation metrics and tracking |
| `src/utils/visualization.py` | ~145 | Plots: confusion matrix, curves, comparisons |
| `src/utils/reproducibility.py` | ~70 | Seed, environment, device utilities |
| `run/pipeline/train.py` | ~50 | Hydra entry point for training |
| `run/pipeline/compare.py` | ~80 | Cross-experiment comparison report |
