# Implementation Verification 2026-03-30

Scope:

- `implementations/drl/halftoning-with-multiagent-drl-implementation/multiagent_drl.py`
- `implementations/drl/halftoning-with-multiagent-drl-implementation/train_multiagent_drl.py`
- `implementations/drl/halftoning-with-multiagent-drl-implementation/optimize_hparams.py`

Paper:

- `algorithm-0/drl/source-papers/halftoning-with-multi-agent-drl-arxiv-2207.11408.pdf`

## Findings

### 1. Objective drift was real

The training code had previously introduced non-paper terms beyond
`L_total = L_MARL + ω_a L_AS`.

Status:

- fixed

Current behavior:

- `ω_s = 0.006`
- `ω_a = 0.002`
- no extra density / entropy / binarization objective terms

### 2. SSIM counterfactual term had a correctness bug

The local-expectation estimator relies on exact per-pixel reward differences
`R(h_a=1, h_-a) - R(h_a=0, h_-a)`.

The HVS-MSE counterfactual term was already exact.

The SSIM counterfactual term was not exact before the fix. The bug was in
how offset-local SSIM-map contributions were accumulated back to the affected
input pixel near image borders.

Status:

- fixed in `_counterfactual_ssim_diff`

### 3. MARL surrogate had an extra spatial averaging bug

The local-expectation gradient should average over the batch, not over
all pixels. The previous code used `.mean()` over the full tensor, which
suppressed the MARL gradient by an extra `H*W` factor.

Status:

- fixed in `local_expectation_policy_loss`

### 4. Even-sized FFT crops were mis-centered in the anisotropy loss

The anisotropy code used `(size - 1) / 2` as the radial center after
`fftshift`. That is wrong for even spatial sizes. On `64x64` crops, the
DC component was incorrectly assigned to nonzero radii, so a perfectly
constant gray field produced a large nonzero anisotropy penalty that grew
with brightness.

Status:

- fixed in `_radial_indices`

## Executable Checks

### Counterfactual exactness

Method:

- brute-force all single-pixel flips on random `5x5` binary halftones
- compare the implementation output to exact recomputation of reward terms

Results after the fix:

- `max_mse_err = 1.16e-09`
- `max_ssim_err = 5.22e-08` for `ssim_kernel_size = 1`
- `max_ssim_err = 1.30e-08` for `ssim_kernel_size = 3`
- `max_ssim_err = 4.47e-08` for `ssim_kernel_size = 5`

Interpretation:

- both counterfactual terms are now exact to numerical precision on the
  tested cases

### Policy-gradient estimator sanity check

Method:

- construct a `2x2` toy problem
- enumerate all `2^4` halftones to compute the exact expected-reward gradient
- compare against the implementation’s Monte Carlo local-expectation gradient
  averaged over `20,000` samples

Observed result:

- `max_abs_err = 4.30e-05`

Interpretation:

- the implemented estimator matches the exact expected-reward gradient on the
  tested toy problem to Monte Carlo tolerance

### Objective-scale diagnostic

Method:

- measure gradient norms of `L_MARL` and `ω_a L_AS` on the same VOC batch
- compare before and after the two objective fixes

Observed result after the fixes:

- `policy_grad_norm = 1.43e-01`
- `anis_grad_norm = 4.66e-08`
- `ratio = 3.26e-07`

Interpretation:

- the anisotropy term no longer hijacks early optimization on constant-gray
  batches
- the previous dark-collapse behavior was caused by implementation bugs,
  not by the paper objective itself

### Unit regression test

Command:

```bash
.venv/bin/python -m unittest implementations/drl/halftoning-with-multiagent-drl-implementation/test_multiagent_drl.py
```

Observed result:

```text
....
----------------------------------------------------------------------
Ran 4 tests in 0.072s

OK
```

### Trainer smoke test

Command:

```bash
env FORCE_COLOR=1 PYTHONUNBUFFERED=1 .venv/bin/python \
  implementations/drl/halftoning-with-multiagent-drl-implementation/train_multiagent_drl.py \
  --dataset-root datasets/bsd/img \
  --eval-root datasets/kodak/img \
  --run-dir /tmp/halftoning-objective-verify \
  --device auto \
  --max-train-images 8 \
  --num-workers 0 \
  --preview-count 1 \
  --dry-run
```

Observed result:

- trainer starts successfully
- objective banner shows `ssim=6.000e-03 | anis=2.000e-03`
- preview generation succeeds

### HPO smoke test

Command:

```bash
env FORCE_COLOR=1 PYTHONUNBUFFERED=1 .venv/bin/python \
  implementations/drl/halftoning-with-multiagent-drl-implementation/optimize_hparams.py \
  --dataset-root datasets/bsd/img \
  --eval-root datasets/kodak/img \
  --run-dir /tmp/halftoning-hpo-verify \
  --device cpu \
  --max-train-images 8 \
  --max-eval-images 1 \
  --num-workers 0 \
  --trials 1 \
  --stage-steps 1 \
  --survivor-fraction 1.0
```

Observed result:

- HPO completes
- only `anisotropy_weight` and `ssim_weight` are searched

## Paper Consistency Notes

- Equation (12) defines `E(h, c) = MSE(G(h), G(c)) - ω_s SSIM(h, c)`.
- Equation (13) defines `L_total = L_MARL + ω_a L_AS`.
- Equation (11) defines `L_AS` as the squared ring-deviation numerator term;
  it is not the fully normalized anisotropy measure from Equation (10).

## VOC Smoke Probe

Probe command:

```bash
env FORCE_COLOR=1 PYTHONUNBUFFERED=1 .venv/bin/python \
  implementations/drl/halftoning-with-multiagent-drl-implementation/train_multiagent_drl.py \
  --dataset-root datasets/voc2012/archive/VOC2012_train_val/VOC2012_train_val/JPEGImages \
  --dataset-manifest implementations/drl/halftoning-with-multiagent-drl-implementation/manifests/voc2012/train.txt \
  --eval-root datasets/voc2012/archive/VOC2012_train_val/VOC2012_train_val/JPEGImages \
  --eval-manifest implementations/drl/halftoning-with-multiagent-drl-implementation/manifests/voc2012/val.txt \
  --run-dir /tmp/voc2012-paper-smoke-50 \
  --device auto \
  --num-workers 0 \
  --preview-count 2 \
  --preview-interval 25 \
  --checkpoint-interval 50 \
  --log-interval 10 \
  --total-iterations 50
```

Observed trend:

- `tone_error` dropped to about `0.0030`
- `ssim` rose to about `0.0697`
- `prob_mean` stayed in a plausible range around `0.41` to `0.43`
- `gray_prob_mean` stayed close to the sampled gray target instead of sinking
- the probability maps tracked image structure instead of collapsing to dark
  noise or contour masks
- the thresholded halftone previews were still blob-like at `50` steps and
  not yet representative of a converged dispersed-dot halftone

Interpretation:

- this is not evidence of convergence
- it is evidence that the repaired implementation now optimizes in the right
  qualitative direction on VOC2012
- a longer run is justified algorithmically, but a full `200000`-step retrain
  is not practical on the current CPU-only machine

## Remaining Limits

- no full post-fix training run has been completed yet
- the existing saved runs in `output/` are pre-fix artifacts and should be
  treated as historical baselines only
- exact paper-result reproduction still depends on matching the paper’s dataset
  split and evaluation protocol
- the paper explicitly specifies `ColorJitter(brightness=0.9)` during training;
  the trainer default has been kept at `0.9` and should remain fixed for
  paper-faithful runs

## Argmax Mismatch Diagnosis

- The stopped VOC run `output/full-voc2012-paper-20260330-144705` is stable
  numerically through step `40000`, but it still fails the paper's constant-gray
  test.
- Direct checkpoint probes show the learned probability fields are reasonable in
  expectation while deterministic `argmax` inference is not:
  - gray `0.25`: `mean_prob=0.248693`, `argmax_mean=0.076263`,
    `sample_mean=0.250519`
  - gray `0.50`: `mean_prob=0.501708`, `argmax_mean=0.513123`,
    `sample_mean=0.499420`
  - gray `0.60`: `mean_prob=0.601242`, `argmax_mean=0.936859`,
    `sample_mean=0.597870`
  - gray `0.75`: `mean_prob=0.750458`, `argmax_mean=0.851807`,
    `sample_mean=0.748535`
- For gray `0.60`, most probabilities land just above the threshold instead of
  forming a dispersed near-binary ranking field:
  - `35.95%` of pixels in `[0.5, 0.6)`
  - `54.79%` of pixels in `[0.6, 0.9)`
- The spectral gap between probability-space and argmax-space is large:
  - gray `0.60` probability map: `anisotropy=0.000136`, `lowfreq_r4=0.005078`
  - gray `0.60` argmax halftone: `anisotropy=0.073766`, `lowfreq_r4=4.947666`
  - gray `0.60` Bernoulli sample: `anisotropy=0.055616`, `lowfreq_r4=0.966773`
- This strongly suggests the remaining mismatch is not a dark-collapse bug.
  It is a training/inference gap: the paper objective is optimized on expected
  Bernoulli behavior and probability-space anisotropy, but our learned policy is
  not sharpening into an argmax-safe dispersed-dot ordering.
