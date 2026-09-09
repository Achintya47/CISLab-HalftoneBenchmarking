# Halftoning with Multi-Agent DRL

This directory contains the runnable extracted implementation.

The original notebook scratchpad that informed the first extraction is not kept
in the working tree. The maintained entrypoints here are the Python modules and
CLIs below.

## Layout

- `multiagent_drl.py`: extracted model, reward, inference, and train-step logic
- `train_multiagent_drl.py`: local training CLI with checkpoints, previews, and pretty terminal logging
- `reports/`: optimization and paper-alignment audit notes
- `output/`: method-local runs and previews

## Training Workflow

Use the registered paper workflow for the reproducible 200,000-step run. It performs a CUDA/data/provenance preflight before training:

```bash
bash scripts/train_marl_paper_200k.sh
```

For a setup-only trainer check, append `--dry-run` to the equivalent `train_multiagent_drl.py --variant paper ...` command. Canonical VOC and Kodak layouts are documented in `docs/DATASETS.md`.

## Hyperparameter Search

Use `optimize_hparams.py` to run successive-halving hyperparameter optimization over the paper-consistent loss weights (`ssim_weight` and `anisotropy_weight`). When `--apply-best` is set, the winning overrides are written to `default_hparams.json`, and `train_multiagent_drl.py` will use them automatically via `reference_config()`.

```bash
python paper-implementations/drl/halftoning-with-multiagent-drl-implementation/optimize_hparams.py --dataset-root datasets/bsds500/BSR/BSDS500/data/images/train --eval-root datasets/kodak/images --run-dir paper-implementations/drl/halftoning-with-multiagent-drl-implementation/output/hpo --device cuda --trials 12 --stage-steps 400,1200,3200 --num-workers 0 --max-eval-images 8 --apply-best
```

After that completes, launch training with `--variant tuned`. The tuned defaults will be picked up unless you override them on the command line. The trainer defaults to `--variant paper`, which disables repository-specific auxiliary losses.
