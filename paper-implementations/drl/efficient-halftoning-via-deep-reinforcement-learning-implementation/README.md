# Efficient Halftoning via Deep Reinforcement Learning

This directory contains a compact reference implementation for the paper:

- Haitian Jiang, Dongliang Xiong, Xiaowen Jiang, Li Ding, Liang Chen, Kai Huang
- *Efficient Halftoning via Deep Reinforcement Learning*
- IEEE Transactions on Image Processing, 2023
- DOI: `10.1109/TIP.2023.3318937`

The implementation now exposes two explicit training variants:

- `stabilized`: the collapse-resistant variant used for the successful BSD run in this repo. It keeps the extra density-consistency term and the simpler REINFORCE-style estimator.
- `paper`: a closer paper-facing path with the local-expectation estimator, paper weights, Näsänen-style HVS filter, and paper CSSIM definition. This is the branch to use when the goal is fidelity to the paper rather than stability tweaks.

## Files

- `efficient_halftoning_drl.py`: compact reference implementation
- `train_efficient_halftoning_drl.py`: training driver with checkpoints, previews, resume support, and `stabilized` / `paper` variants
- `efficient-halftoning-via-deep-reinforcement-learning-review.md`: mismatch review of the original notebook

## Notes

- The paper specifies a 16-block ResNet with 32 channels and one-step RL training. This implementation follows that.
- The `paper` variant uses a Näsänen-style `11x11` HVS kernel with `S = 2000`, paper CSSIM weighting, and the local-expectation estimator.
- The `stabilized` variant intentionally keeps the extra density-consistency loss because it prevents the collapse mode observed in the earlier BSD runs. It should not be presented as the paper-faithful path.
- The original notebook in this folder is exploratory and Colab-specific. The Python module is the intended reference going forward.
- The training driver keeps the paper-facing defaults: random `64x64` grayscale crops, batch size `64`, Adam, cosine LR decay, and one-step `0.5`-threshold inference for previews.
- Use `--dataset-manifest` and `--eval-manifest` when you have the exact VOC2012 split lists available. This repo does not bundle the paper’s VOC split files.
- Full training is still intentionally separate from the audit pass; the minimal demo path remains a no-training forward pass for interface validation.

## Example Commands

Stabilized training:

```bash
.venv/bin/python paper-implementations/drl/efficient-halftoning-via-deep-reinforcement-learning-implementation/train_efficient_halftoning_drl.py \
  --variant stabilized \
  --dataset-root datasets/bsd/img \
  --eval-root datasets/kodak/img \
  --run-dir paper-implementations/drl/efficient-halftoning-via-deep-reinforcement-learning-implementation/output/stabilized-run
```

Paper-oriented training once VOC images and split manifests are available:

```bash
.venv/bin/python paper-implementations/drl/efficient-halftoning-via-deep-reinforcement-learning-implementation/train_efficient_halftoning_drl.py \
  --variant paper \
  --dataset-root /path/to/VOC2012-grayscale \
  --dataset-manifest /path/to/voc_train_manifest.txt \
  --eval-root /path/to/VOC2012-grayscale \
  --eval-manifest /path/to/voc_val_manifest.txt \
  --run-dir paper-implementations/drl/efficient-halftoning-via-deep-reinforcement-learning-implementation/output/paper-run
```
