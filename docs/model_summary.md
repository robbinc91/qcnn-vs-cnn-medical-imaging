# Model Architecture Summary

All models trained on the brainstem segmentation dataset (4 classes: background, medulla, pons, mesencephalon).
Input: **80×80** (2D axial slices) / **80×80×96** (3D volumes).
Depth 4 → bottleneck resolution: **5×5** (2D) / **5×5×6** (3D).

## Architecture Comparison Table

| Model | Dims | Params | Depth (pools) | Channels | Block type | Norm | Activation | Skip connections | Bottleneck |
|-------|------|-------:|:-------------:|----------|------------|------|------------|-----------------|------------|
| ClassicalCNN | 2D | 16.95M | 4 | [64, 128, 256, 352] | DoubleConv | BatchNorm | ReLU | concat | DoubleConv |
| ClassicalCNN | 3D | 16.95M | 4 | [38, 76, 156, 204] | DoubleConv | BatchNorm | ReLU | concat | DoubleConv |
| CerebNet | 2D | 16.65M | 4 | 160 (uniform) | CompetitiveDense (5×5) | BatchNorm | PReLU | MaxUnpool + competitive max | CompetitiveDense |
| CerebNet | 3D | 17.98M | 4 | 160 (uniform) | CompetitiveDense (3×3×3) | BatchNorm | PReLU | MaxUnpool + competitive max | CompetitiveDense |
| MipaimUNet | 2D | 17.27M | 4 | [90, 180, 360, 720, 1440] | Inception multi-branch | InstanceNorm | ReLU | attention refinement | Inception |
| MipaimUNet | 3D | 17.80M | 4 | [64, 128, 256, 512, 1024] | Inception multi-branch | InstanceNorm | ReLU | attention refinement | Inception |
| SwinUNETR | 2D | 16.79M | 4 | fs=36, heads=[3,6,12,24] | Swin Transformer | LayerNorm | GELU | concat | Swin block |
| SwinUNETR | 3D | 17.06M | 4 | fs=24, heads=[3,6,12,24] | Swin Transformer | LayerNorm | GELU | concat | Swin block |
| ACAPULCO | 2D | 17.38M | 4 | [44, 88, 176, 240] | PreActResBlock | InstanceNorm | ReLU (pre-act) | concat | 2× PreActResBlock |
| ACAPULCO | 3D | 17.08M | 4 | [26, 52, 104, 140] | PreActResBlock | InstanceNorm | ReLU (pre-act) | concat | 2× PreActResBlock |
| MARIN | 2D | 17.42M | 4 | [44, 88, 176, 240] | MultiScale PreAct (dil=1‖dil=2) | InstanceNorm | PReLU | CBAM (ch+spatial attn) | Competitive max + MultiScale |
| MARIN | 3D | 17.10M | 4 | [26, 52, 104, 140] | MultiScale PreAct (dil=1‖dil=2) | InstanceNorm | PReLU | CBAM (ch+spatial attn) | Competitive max + MultiScale |
| QCNN PennyLane | 2D | 6.09M† | 3 | [80,160,320] | DoubleConv U-Net + additive quantum branch | BatchNorm | ReLU | concat | global quantum recalibration (additive) |
| QCNN PennyLane | 3D | 17.59M | 3 | [80,160,320] | DoubleConv U-Net + additive quantum branch | BatchNorm | ReLU | concat | global quantum recalibration (additive) |
| QCNN Qiskit | 2D | 6.09M† | 3 | [80,160,320] | DoubleConv U-Net + additive quantum branch | BatchNorm | ReLU | concat | global quantum recalibration (additive) |
| QCNN Qiskit | 3D | 17.58M | 3 | [80,160,320] | DoubleConv U-Net + additive quantum branch | BatchNorm | ReLU | concat | global quantum recalibration (additive) |
| Hybrid QCNN | 2D | 6.09M† | 3 | [80,160,320] | DoubleConv U-Net + additive quantum branch | BatchNorm | ReLU | concat | global quantum recalibration (additive) |
| Hybrid QCNN | 3D | 17.59M | 3 | [80,160,320] | DoubleConv U-Net + additive quantum branch | BatchNorm | ReLU | concat | global quantum recalibration (additive) |

†Native 2D configs are ~6.09M (3 encoder stages), i.e. ~⅓ of the ~17M classical field — so the *native* 2D comparison is not parameter-matched. The parameter-matched 2D runs add a 4th encoder stage `[80,160,320,480]` → 16.90M (see `run/rerun_qcnn_2d_17M.sh`).

## Notes

- **MipaimUNet** has 5 encoder levels **and** an explicit bottleneck block (`bottleneck_block`, applied to the deepest features before upsampling); all models have an explicit bottleneck transform.
- **CerebNet** uses index-based MaxPool/MaxUnpool instead of ConvTranspose for upsampling. It **does** normalize: every conv is Conv→BatchNorm→PReLU (not "no normalization / ReLU").
- **MARIN**'s dual-dilation block splits `out_ch` into two parallel branches (dil=1 and dil=2, each `out_ch//2` channels), giving a 3×3 and effective 5×5 receptive field. The two branches together match the cost of a single 3×3 conv, but each block then adds a full `out_ch→out_ch` `integrate` conv, so a `MultiScalePreActBlock` costs **roughly 2× a single 3×3 conv**, not the same.
- **MARIN** is the only model combining pre-activation ordering, learnable PReLU, dual-dilation multi-scale branches, CBAM attention on skip connections, and competitive maxout at the bottleneck.
- **SwinUNETR** is the only non-convolutional encoder; uses window-based self-attention (window size 7).
- The three **quantum/hybrid models are all hybrid CNNs**: a classical conv U-Net whose deepest features are *added* to a global, spatially-uniform vector produced by a quantum circuit (`x = x + q_feat`). The quantum branch is a global channel-recalibration side-branch, not a bottleneck the data flows through, and it does **not** replace conv layers. The smaller native-2D parameter counts come from the **narrower 3-stage conv encoder** `[80,160,320]` (vs the 4-stage ~17M classical encoders), not from the quantum layer; the quantum branch itself adds negligible parameters.
