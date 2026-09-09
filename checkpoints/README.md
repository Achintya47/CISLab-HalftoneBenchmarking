# Benchmark checkpoints

Checkpoint binaries are distributed through versioned GitHub Releases. The JSON files in this directory are the trusted metadata records used by `python -m benchmarking.checkpoints`.

`marl-paper-v1` remains explicitly pending until a CUDA machine completes the registered 200,000-step run and the quality gates in `benchmarking.marl_release` pass. A pending record is never downloaded or silently replaced with an untrained model.
