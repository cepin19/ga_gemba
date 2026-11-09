# Shared settings for the LREC replication. Models live in the llm_agents cache.
REPO=/lnet/work/people/jon/ga_gemba
OLD=/lnet/work/people/jon/ga_clean_comet-v2
RUNS=$REPO/runs/lrec
BLEURT_CKPT=$OLD/bleurt/checkpoints/BLEURT-20/
UNITE_MODEL=/lnet/work/people/jon/llm_agents/.cache/modelscope/unite_mup
PY=/lnet/work/people/jon/comet2/bin/python
export HOME=/lnet/work/people/jon/llm_agents
export PYTHONPATH=$REPO:$RUNS/overlay
export TOKENIZERS_PARALLELISM=false TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=/opt/cuda/11.2/lib64:/opt/cuda/11.2/cudnn/8.1.1/lib64:/opt/cuda/11.2/nccl/2.8.4/lib:/opt/cuda/11.2/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}
export UNITE_MODEL
