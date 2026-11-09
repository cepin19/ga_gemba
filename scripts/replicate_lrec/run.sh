#!/bin/bash
#SBATCH -J lrec-ga
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpuram48G|gpuram40G|gpuram24G
#SBATCH --mem=30G
#SBATCH --cpus-per-task=2
#
# One line of matrix.py. Array task id = line number. DATA_DIR holds src, hyps, ref, RUN_DIR the outputs.
#   sbatch --array=0-48%12 -p gpu-ms,gpu-troja run.sh DATA_DIR RUN_DIR [extra ga-gemba args]
set -euo pipefail
source /lnet/work/people/jon/ga_gemba/scripts/replicate_lrec/env.sh
data=$1; run_dir=$2; shift 2
mkdir -p "$run_dir"
line=$($PY "$REPO/scripts/replicate_lrec/matrix.py" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
eval "set -- $line \"\$@\""
name=$1; shift
echo "$name on $(hostname), start $(date +%s)"
exec $PY -m ga_gemba adversarial -s "$data/src.txt" -t "$data/hyps.txt" -r "$data/ref.txt" --num-samples 40 \
    -w "$OLD/english_wordlist.txt" -g 149 -m 1 -c 0.1 --pop-multiplier 50 --max-len-factor 1.1 \
    --no-bleu-effective-order --bleurt-checkpoint "$BLEURT_CKPT" --unite-model "$UNITE_MODEL" --resume \
    -o "$run_dir/$name.txt" "$@"
