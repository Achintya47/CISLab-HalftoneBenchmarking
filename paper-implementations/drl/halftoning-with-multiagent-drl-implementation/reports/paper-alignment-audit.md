# Paper Alignment Audit

Paper: `algorithm-0/drl/source-papers/halftoning-with-multi-agent-drl-arxiv-2207.11408.pdf`

Code audited:
- `implementations/drl/halftoning-with-multiagent-drl-implementation/multiagent_drl.py`
- `implementations/drl/halftoning-with-multiagent-drl-implementation/train_multiagent_drl.py`

## Core Alignment

- Shared one-step multi-agent policy over pixels: matched.
- Backbone shape: `16` residual blocks, `32` channels: matched.
- State formulation: continuous-tone image plus sampled Gaussian noise: matched.
- Objective form: `L_total = L_MARL + ω_a L_AS`: matched.
- Paper weights: `ω_s = 0.006`, `ω_a = 0.002`: matched.
- HVS term: Näsänen-style low-pass kernel with moderate scale `2000`: matched in implementation.
- Structural term: standard SSIM term in the reward: matched.
- Estimator: local-expectation / COMA-style counterfactual action summation instead of REINFORCE baseline: matched, with the SSIM counterfactual term numerically checked against exact pixel-flip enumeration.
- MARL surrogate scaling: batch-mean over per-pixel counterfactual sums, without an extra spatial averaging factor: matched after fix.
- Anisotropy loss: computed on raw probabilities for constant-gray batches and implemented as the numerator-only squared ring deviation described by Equation (11): matched.
- FFT centering for even crop sizes: matched after fixing the radial center to the `fftshift` midpoint.
- Constant-gray anisotropy batch: uniformly sampled scalar gray levels expanded spatially each iteration: matched.
- Inference rule: discrete `argmax`, equivalent here to threshold `0.5`: matched.
- Training defaults: `200000` iterations, batch `64`, Adam, cosine annealing `3e-4 -> 1e-5`, brightness jitter `0.9`: matched.

## Remaining Reproduction Caveats

- No public official HALFTONERS implementation was located during audit. As of
  2026-03-30, CatalyzeX still lists the paper as `Request Code`, so there is no
  public reference implementation to inspect for undocumented low-frequency
  penalties, auxiliary terms, or inference-time heuristics.
- The trainer is still dataset-agnostic. Exact paper reproduction still requires supplying the paper’s VOC2012 split manifests externally.
- Preview image selection is operational rather than paper-benchmark-specific unless explicit manifests are provided.

## Conclusion

After restoring the paper objective and correcting the SSIM counterfactual computation, the MARL surrogate scaling, and the even-size FFT centering bug in the anisotropy loss, no remaining high-severity algorithmic mismatch was identified in the audited scope. Exact paper-result reproduction still depends on using the correct VOC2012 split, evaluation protocol, and suitable training hardware.
