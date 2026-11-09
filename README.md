# ga-gemba

A genetic algorithm over candidate translations, with one core and two uses:

- **`mbr`** improves translation quality: it evolves candidates towards agreement with
  pseudo-references (by default, the candidates themselves).
- **`adversarial`** generates adversarial examples for metrics: it finds translations that one
  metric rates highly while others rate them poorly (or the reverse).

Any metric can be used in either mode, including **GEMBA**, an LLM acting as judge, through a
local vLLM server or a remote OpenAI-style API.

## Install

```bash
pip install -e .              # BLEU, chrF
pip install -e '.[spm]'       # spBLEU and --tokenizer sp
pip install -e '.[gemba]'     # GEMBA
pip install -e '.[comet]'     # COMET
pip install -e '.[test]' && PYTHONPATH=tests pytest
```

BLEURT and UniTE need the `bleurt` and `modelscope` packages. Upstream GEMBA methods (see below)
need the `gemba` package installed from its source. Only the dependencies of the metrics you
select are imported. vLLM may live in another environment: point `--vllm-bin` at it.

## Input

| Option | Content |
|---|---|
| `-s` | one source per line |
| `-t` | `--num-samples` candidates per source, grouped by source |
| `-r` (adversarial) | `--num-refs` true references per source |
| `-p` (mbr, optional) | `--num-pseudo-refs` pseudo-references per source; defaults to `-t` |
| `-d` (optional) | per-source mutation phrases, `;`-separated, one line per source |
| `-w` (optional) | mutation phrases shared by all sources, `;`-separated |

To interleave several translation files into `-t`: `paste -d '\n' sys1.txt sys2.txt sys3.txt`.

## Improving translations

```bash
ga-gemba mbr -s src.txt -t hyps.txt --num-samples 40 -o out.txt \
    --weights comet=0.4,comet_qe=0.4,bleu_multiref=0.1,chrf_multiref=0.1

# spBLEU on subword genes, without a GPU
ga-gemba mbr -s src.txt -t hyps.txt --num-samples 45 -o out.txt \
    --weights spbleu=1 --tokenizer sp -g 200 -m 2 -c 0.2

# only rerank the candidates, no breeding
ga-gemba mbr -s src.txt -t hyps.txt --num-samples 45 -o out.txt --weights spbleu=1 -g 0
```

## Generating adversarial examples

`--target` is the metric to fool; each `--penalize` metric gets weight `-penalty`. With
`--direction minimize`, all signs flip: the target is pushed down while the guards are pushed up.
Use `--weights` instead for any other combination.

```bash
ga-gemba adversarial -s src.txt -t hyps.txt -r ref.txt --num-samples 40 -o adv.txt \
    -w wordlist.txt --target unite --penalize bleu_multiref,chrf_multiref,comet,comet_qe --penalty 0.1
```

The JSONL output (`OUTPUT.jsonl`, or `--jsonl`) holds per-metric scores for the best initial
candidate and for the result. `--report-metrics a,b` also records metrics that are not optimised,
which shows whether an attack transfers to them.

## GEMBA

| Metric | Score |
|---|---|
| `gemba_accuracy` | accuracy prompt |
| `gemba_dialect` | dialect authenticity prompt |
| `gemba` | mean of both for dual methods, otherwise the method's single score |

`--gemba-method` selects `GEMBA-DA` (default) or `GEMBA-DA_ref`, which are the dual
accuracy + dialect prompts, or an upstream method: `GEMBA-MQM`, `GEMBA-SQM[_ref]`,
`GEMBA-stars[_ref]`, `GEMBA-classes[_ref]`. Reference methods average over all references, so with
pseudo-references the cost is multiplied by their number. Scores are scaled to 0-1.

`wmt27-DA_accuracy`, `wmt27-DA_dialect` and `wmt27-MQM` take the prompt and parser from
`score_gemba.py` of the wmt27 experiments (`--gemba-wmt27-scripts`), imported rather than copied, and
use the metric `gemba`. Its accuracy prompt is the same text as `GEMBA-DA` accuracy, but its parser
only accepts `Score: N`, and as there each prompt is asked once (`--gemba-parse-retries 1`) with
`--gemba-max-tokens 2048`. The source language comes from `--src-lang-name`, where `score_gemba.py`
fixes it to English. MQM penalties (-25 to 0) are scaled to 0-1.

Prompts need language names: `--src-lang-name English --tgt-lang-name "Egyptian Arabic"`, or
`--tgt-lang-file` with one name per source line for multi-dialect data.

```bash
# vLLM started and stopped by the run
ga-gemba adversarial -s src.txt -t hyps.txt -r ref.txt --num-samples 40 -o adv.txt -w wordlist.txt \
    --target gemba_accuracy --penalize chrf,bleu \
    --gemba-backend vllm --gemba-model google/gemma-4-31B-it \
    --vllm-bin /path/to/env/bin/vllm --vllm-gpus 2 --vllm-max-model-len 40000 \
    --vllm-arg=--max-num-seqs=16 \
    --src-lang-name English --tgt-lang-name "Egyptian Arabic"

# a vLLM server that is already running (for example shared by several shards)
ga-gemba mbr ... --weights gemba=1 --gemba-model google/gemma-4-31B-it \
    --gemba-base-url http://gpu-node:5005/v1 --gemba-max-requests -1

# remote OpenAI-style API
export OPENAI_API_KEY=...
ga-gemba adversarial ... --target gemba --gemba-model gpt-5.4-nano \
    --gemba-base-url https://api.openai.com/v1 --gemba-max-requests 200000
```

**Model storage.** A vLLM server started by a run gets `HOME=/lnet/work/people/jon/llm_agents`, so
models are stored in its `.cache` rather than `~/.cache` (`--vllm-home` changes this). `HF_HOME` and
similar variables are cleared for that process, and the `HOME` of the GA itself is unchanged. Start
your own servers the same way: `HOME=/lnet/work/people/jon/llm_agents vllm serve ...`.

**Cost.** A run can evaluate up to `population x (generations + 1) x prompts` candidates per
segment; only distinct texts are sent, once each. Non-local endpoints therefore require
`--gemba-max-requests` (`-1` disables the limit). Reaching the limit stops the run with an error;
finished segments are kept, and `--resume` continues from them.

**Temperature** defaults to 0. Selecting the maximum over noisy scores would favour lucky samples
rather than better translations. Retries after an unparsable answer use at least 0.7, since asking
again at temperature 0 would return the same answer.

**Failures.** A candidate the judge cannot score gets fitness -inf and cannot win. Connection and
authentication failures, or 50 consecutive prompts without a usable score, abort the run instead of
producing meaningless output.

**API key.** The variable named by `--gemba-api-key-env` (default `OPENAI_API_KEY`) is sent to
whatever `--gemba-base-url` is set, so check the URL when that variable holds a real key.

Reasoning models may need more than the default `--gemba-max-tokens 500`. Server-specific options
go through `--gemba-extra-body '{"...": ...}'`.

## Running on SLURM

Shards are selected with `--shard I/N`, so no input splitting is needed. Resubmitted tasks resume.

```bash
GA_ENV=/path/to/venv sbatch --array=0-39%10 -p cpu-troja scripts/slurm_shard.sh runs/exp1 mbr \
    -s src.txt -t hyps.txt --num-samples 45 --weights spbleu=1 --tokenizer sp -g 200 -m 2 -c 0.2

ga-gemba collect runs/exp1/shard_*.jsonl -o runs/exp1/out.txt --merged-jsonl runs/exp1/out.jsonl
```

`collect` refuses incomplete runs unless given `--allow-partial`, and prints per-metric means.

## Migrating from `comet_mbr.py` / `comet_mbr_adv.py`

| Old | New |
|---|---|
| `-f fitness_comet_mbr_and_qe_and_bleu_and_chrf_w_multiref --comet_mbr_w=.4 --comet_qe_w=.4 --bleu_w=.1 --chrf_w=.1` | `mbr --weights comet=.4,comet_qe=.4,bleu_multiref=.1,chrf_multiref=.1` |
| `-f fitness_spbleu` / `fitness_spbleu_multiref` | `--weights spbleu=1` / `spbleu_multiref=1` |
| `-f fitness_ref_combined_multiref --unite_w=1 --comet_mbr_w=-.1 ...`, `--num_pseudo_refs 1` | `adversarial -r ref.txt --target unite --penalize comet,... --penalty .1` |
| `--model`, `--model-qe` | `--comet-model`, `--comet-qe-model` |
| `-l log`, `.out_head+N` files, chunk scripts | `--jsonl`, `--history`, `--shard`, `--resume` |
| population hard-coded to x15 (MBR) or x50 (adversarial) | `--pop-multiplier`, same defaults |

### Deliberate differences

These bugs in the old code are fixed, so results are not directly comparable with earlier runs:

- **chrF replaced by BLEU.** `solution.score` was shared between metrics, so in combined fitness
  functions that ran the sacreBLEU metrics in turn (e.g. `fitness_comet_mbr_and_qe_and_bleu_and_chrf_w[_multiref]`
  in `comet_mbr.py`), the chrF term silently received the BLEU value.
- **`-g 0` returned the first candidate** without scoring anything. It now reranks.
- **Score cache shared across sources.** `score_cache` was keyed by hypothesis text only, so a
  repeated text could reuse a score computed for another source. The memo is now per segment.
- **Identical pseudo-references were not excluded in practice:** first-generation text kept its
  trailing newline, so it never equalled a stripped reference. Exclusion now works and depends on
  the text rather than lineage, so originals and offspring are treated alike. It applies to
  BLEU-family metrics, chrF and GEMBA `_ref` methods; COMET still compares with all references, as
  before. Use `--keep-identical-refs` to disable it.
- **Crossover crashed** on chromosomes shorter than 4 genes.
- **`--no-empty` still deleted genes.** It now really disables deletion.
- **Dictionary entries in `--tokenizer sp`** produced double spaces.

Other changes:

- The defaults of `adversarial` follow the LREC 2024 code except `-g`: use `-g 149` to evaluate the
  same 150 populations, and `--no-bleu-effective-order` for its BLEU fitness. See
  `scripts/replicate_lrec/`.
- `-g N` evaluates the initial population plus N bred generations; the old loop evaluated N
  populations in total.
- Runs are seeded (`--seed`), per segment, so shards give the same result as one process.
- Mismatched line counts are errors, and existing outputs are not overwritten without `--resume`.
- GEMBA `GEMBA-DA` here means the dual accuracy + dialect prompts from `aramt`, not the single-score
  prompt of that name in upstream GEMBA. Unlike `aramt`, the default temperature is 0 and the parser
  also accepts bare numbers and Markdown such as `**Score:** 85`.

## Layout

```
ga_gemba/ga.py          the algorithm; fitness is an opaque callable
ga_gemba/objective.py   weighted metrics and per-segment memo
ga_gemba/metrics/       sacre (BLEU family, chrF), comet, neural (BLEURT, UniTE), gemba
ga_gemba/vllm_server.py managed vLLM process
ga_gemba/cli.py         mbr, adversarial, collect
scripts/slurm_shard.sh  array-job wrapper
```
