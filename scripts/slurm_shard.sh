#!/bin/bash
#SBATCH -J ga-gemba
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#
# One shard of a GA run, as a SLURM array task. A resubmitted task resumes.
#
#   sbatch --array=0-39%10 -p gpu-ms --gres=gpu:1 scripts/slurm_shard.sh RUN_DIR mbr \
#       -s src.txt -t hyps.txt --num-samples 40 --weights spbleu=1
#
# Set GA_ENV to activate a Python environment; run sbatch from the repository root.
set -euo pipefail

run_dir=$1
command=$2
shift 2

if [[ -n "${GA_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "$GA_ENV/bin/activate"
fi
export TOKENIZERS_PARALLELISM=false

shard=$(printf '%04d' "$SLURM_ARRAY_TASK_ID")
mkdir -p "$run_dir"
exec python -m ga_gemba "$command" "$@" \
    --shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT" \
    --resume \
    --vllm-log "$run_dir/vllm_$shard.log" \
    -o "$run_dir/shard_$shard.txt" \
    --jsonl "$run_dir/shard_$shard.jsonl"
