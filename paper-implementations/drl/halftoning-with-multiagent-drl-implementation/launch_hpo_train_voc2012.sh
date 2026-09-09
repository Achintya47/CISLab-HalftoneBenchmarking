#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "$repo_root"

ts="${1:-$(date +%Y%m%d-%H%M%S)}"
hpo_run_dir="paper-implementations/drl/halftoning-with-multiagent-drl-implementation/output/hpo-voc2012-nonpaper-${ts}"

# Output directory labeled as "paper" when invocation uses "tuned", thus change the directory name to "tuned"
train_run_dir="paper-implementations/drl/halftoning-with-multiagent-drl-implementation/output/full-voc2012-tuned-hpo-200k-${ts}"

env FORCE_COLOR=1 PYTHONUNBUFFERED=1 .venv/bin/python \
  paper-implementations/drl/halftoning-with-multiagent-drl-implementation/optimize_hparams.py \
  --dataset-root datasets/voc2012/VOCdevkit/VOC2012/JPEGImages \
  --dataset-manifest paper-implementations/drl/halftoning-with-multiagent-drl-implementation/manifests/voc2012/train.txt \
  --eval-root datasets/voc2012/VOCdevkit/VOC2012/JPEGImages \
  --eval-manifest paper-implementations/drl/halftoning-with-multiagent-drl-implementation/manifests/voc2012/val.txt \
  --run-dir "$hpo_run_dir" \
  --device cuda \
  --num-workers 0 \
  --trials 12 \
  --stage-steps 2000,8000,20000 \
  --survivor-fraction 0.5 \
  --max-eval-images 8 \
  --search-space nonpaper_impl \
  --apply-best

env FORCE_COLOR=1 PYTHONUNBUFFERED=1 .venv/bin/python \
  paper-implementations/drl/halftoning-with-multiagent-drl-implementation/train_multiagent_drl.py \
  --variant tuned \
  --dataset-root datasets/voc2012/VOCdevkit/VOC2012/JPEGImages \
  --dataset-manifest paper-implementations/drl/halftoning-with-multiagent-drl-implementation/manifests/voc2012/train.txt \
  --eval-root datasets/voc2012/VOCdevkit/VOC2012/JPEGImages \
  --eval-manifest paper-implementations/drl/halftoning-with-multiagent-drl-implementation/manifests/voc2012/val.txt \
  --run-dir "$train_run_dir" \
  --device cuda \
  --num-workers 4 \
  --preview-count 4 \
  --preview-interval 5000 \
  --checkpoint-interval 5000 \
  --log-interval 100 \
  --total-iterations 200000
