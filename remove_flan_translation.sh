#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LOCAL DATA-PIPELINE MODIFICATION (not upstream sapientinc behaviour)
#
# Remove the FLAN translation (wmt16_translate_*) tasks from the tokenized
# sampling pool. Translation is ~32% of FLAN (~6.2B tokens across 7 task files)
# and is near-useless for English reasoning/knowledge benchmarks (it teaches
# de/fi/cs<->en translation, not MCQA/QA/NLI). Removing it frees that budget for
# the benchmark-relevant non-translation FLAN (see prefix_config.yaml: flan__
# cap raised 5k->15k).
#
# Mechanism: move the matching tokenized dirs out of the pool root. The sampler
# (sample_tokenized.py) only iterates dirs under $TOK, so moved dirs vanish from
# the mix. Idempotent. Reversible with --restore.
#
# NOTE: this is enforced at the *tokenized* level. A future incremental re-run
# of the Rust tokenizer will regenerate these dirs from the cleaned source; just
# re-run this script afterwards (or delete the matching cleaned files to make it
# permanent).
# ---------------------------------------------------------------------------
set -euo pipefail
shopt -s nullglob

TOK="${TOK:-/scratch/data_io_run/data_tokenized_bpe_65k}"
HOLD="${HOLD:-/scratch/data_io_run/_removed_translation}"
mkdir -p "$HOLD"

if [[ "${1:-}" == "--restore" ]]; then
  n=0
  for d in "$HOLD"/*translate*; do mv "$d" "$TOK"/ && n=$((n+1)); done
  echo "restored $n translation dir(s) back into the pool: $TOK"
  exit 0
fi

n=0
for d in "$TOK"/*translate*; do mv "$d" "$HOLD"/ && n=$((n+1)); done
echo "moved $n FLAN translation dir(s) out of the sampling pool -> $HOLD"
echo "(re-run with --restore to undo)"
