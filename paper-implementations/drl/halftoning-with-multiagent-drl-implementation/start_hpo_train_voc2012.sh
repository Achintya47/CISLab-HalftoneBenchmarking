#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "$repo_root"

ts="$(date +%Y%m%d-%H%M%S)"
log="paper-implementations/drl/halftoning-with-multiagent-drl-implementation/output/hpo-train-${ts}.log"
pid_file="paper-implementations/drl/halftoning-with-multiagent-drl-implementation/output/hpo-train-${ts}.pid"

# Avoid first run failure, bash's redirection > doesn't create missing parent directories
# thus this redirection fails on the first run before python's run_dir.mkdir runs.
mkdir -p "$(dirname "$log")"

nohup bash "paper-implementations/drl/halftoning-with-multiagent-drl-implementation/launch_hpo_train_voc2012.sh" "$ts" >"$log" 2>&1 < /dev/null &
pid="$!"
printf '%s\n' "$pid" >"$pid_file"

printf 'pid=%s\n' "$pid"
printf 'ts=%s\n' "$ts"
printf 'log=%s\n' "$log"
printf 'pid_file=%s\n' "$pid_file"