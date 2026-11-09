#!/bin/bash
#SBATCH -J lrec-gemba
#SBATCH -p gpu-ms,gpu-troja
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpuram48G|gpuram40G
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=21-00:00:00
#
# The LREC setup (data, candidates, wordlist, GA settings, controls) with a GEMBA judge as the target:
# wmt27 DA accuracy prompt and parser (score_gemba.py), served by vLLM as in the wmt27 runs.
# With two GPUs, GPU 1 runs vLLM and GPU 0 the control metrics (COMET, CometKiwi, UniTE). With one
# GPU (sbatch --gres=gpu:1, 48 GB), both share it and vLLM gets VLLM_MEM (default 0.62) of its memory.
#   sbatch run_gemba.sh DATA_DIR OUT_DIR PENALTY [MODEL [WEIGHTS_DIR [extra ga-gemba args]]]
# WEIGHTS_DIR, if given, is served under the name MODEL (a local copy of an incomplete cache snapshot).
# In an array job, SHARD_COUNT=N runs shard SLURM_ARRAY_TASK_ID of N.
set -euo pipefail
data=$1; out=$2; penalty=$3; model=${4:-google/gemma-4-12B-it}; weights=${5:-$model}
shift $(( $# < 5 ? $# : 5 ))
mkdir -p "$out"
name=GEMBA-acc-${model##*/}_w$penalty
shard=()
if [ -n "${SHARD_COUNT:-}" ]; then
    shard=(--shard "$SLURM_ARRAY_TASK_ID/$SHARD_COUNT")
    name=$name.shard$SLURM_ARRAY_TASK_ID
fi
port=$((20000 + RANDOM % 20000))
IFS=, read -ra gpus <<<"$CUDA_VISIBLE_DEVICES"
if [ ${#gpus[@]} -ge 2 ]; then vllm_gpu=${gpus[1]}; mem=0.9; else vllm_gpu=${gpus[0]}; mem=${VLLM_MEM:-0.62}; fi
echo "$name on $(hostname), GPUs ${CUDA_VISIBLE_DEVICES}, vLLM memory $mem, start $(date +%s)"

# vLLM settings of wmt27 score_gemba_local.py (32k context, 0.9 memory, offline hub).
CUDA_VISIBLE_DEVICES=$vllm_gpu HOME=/lnet/work/people/jon/llm_agents HF_HUB_OFFLINE=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 /lnet/work/people/jon/predict_split_points/env_vllm/bin/vllm serve "$weights" \
    --served-model-name "$model" --host 127.0.0.1 --port $port --max-model-len 32768 --gpu-memory-utilization $mem \
    >"$out/$name.vllm.log" 2>&1 &
vllm_pid=$!
trap 'kill $vllm_pid 2>/dev/null; wait $vllm_pid 2>/dev/null' EXIT
until curl -sf --noproxy '*' http://127.0.0.1:$port/v1/models >/dev/null; do
    kill -0 $vllm_pid 2>/dev/null || { echo "vLLM died, see $out/$name.vllm.log"; exit 1; }
    sleep 10
done
echo "vLLM ready $(date +%s)"

source /lnet/work/people/jon/ga_gemba/scripts/replicate_lrec/env.sh
export PYTHONPATH=$PYTHONPATH:$RUNS/overlay_gemba
export CUDA_VISIBLE_DEVICES=${gpus[0]}
$PY -m ga_gemba adversarial -s "$data/src.txt" -t "$data/hyps.txt" -r "$data/ref.txt" --num-samples 40 \
    -w "$OLD/english_wordlist.txt" -g 149 -m 1 -c 0.1 --pop-multiplier 50 --max-len-factor 1.1 \
    --no-bleu-effective-order --unite-model "$UNITE_MODEL" --resume \
    --target gemba --penalty "$penalty" --penalize chrf_multiref,bleu_multiref,comet,comet_qe,unite \
    --gemba-method wmt27-DA_accuracy --gemba-model "$model" --gemba-base-url http://127.0.0.1:$port/v1 \
    --gemba-concurrency 128 --src-lang-name Czech --tgt-lang-name English \
    "${shard[@]}" -o "$out/$name.txt" "$@"
