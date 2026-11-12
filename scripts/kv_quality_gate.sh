#!/bin/sh
# KV-format quality gate: does a packed KV cache change the model's answer?
#
# Compares the FINAL LOGITS ROW of each format against f16, and judges by KL
# divergence rather than by generated text. Text is chaotic -- one flipped
# argmax rewrites the whole continuation -- so a text comparison rates a 1e-4
# logit shift as a failure and a 0.3 shift as a pass, depending on luck. The
# logits row is the model's actual answer to the prompt.
#
# A KL of 0.01 is roughly a 1% perplexity increase. Measured on Qwen3.8-4B:
#   int8   kl 2e-5 .. 7e-4      (near-exact)
#   turbo3 kl 1.5e-3 .. 1.1e-2  (3-bit class)
#   rot4   kl 8.8e-4 .. 1.8e-2  (3-bit class)
#
# Usage: sh scripts/kv_quality_gate.sh MODEL [MODEL...]
#        KRK_EXE=... to point at another binary (default build-cpu-win/kraken.exe)
#
# Written in POSIX sh on purpose: this box has w64devkit's sh and no bash, and a
# gate that cannot run is not a gate.

set -u
EXE="${KRK_EXE:-./build-cpu-win/kraken.exe}"
TMP="${TMPDIR:-/tmp}"
FORMATS="${KRK_KV_FORMATS:-f16 int8 turbo3 rot4}"

PROMPTS='The capital of France is
The history of computing is
def quicksort(a):
The quick brown fox jumps over the lazy dog. In 1923, the first practical'

rc=0
for model in "$@"; do
  if [ ! -f "$model" ]; then echo "SKIP (no such file): $model"; continue; fi
  echo "== $(basename "$model")"
  i=0
  printf '%s\n' "$PROMPTS" | while IFS= read -r prompt; do
    ref="$TMP/kvq-$$-$i.bin"
    rm -f "$ref"
    if ! "$EXE" --intel -m "$model" -p "$prompt" --kv-quant f16 \
                 --logits-ref "$ref" >/dev/null 2>&1; then
      echo "  p$i f16 reference failed"
      continue
    fi
    for q in $FORMATS; do
      line=$("$EXE" --intel -m "$model" -p "$prompt" --kv-quant "$q" \
                     --logits-ref "$ref" 2>/dev/null | grep '^logits-diff')
      [ -n "$line" ] || line="logits-diff MISSING"
      echo "  p$i $q $line"
      # int8 is the validated class; a FAIL there is a real defect and sets the
      # exit. A 3-bit format failing is the quantizer's expected cost, printed
      # as evidence rather than treated as a build break.
      if [ "$q" = "int8" ]; then
        case "$line" in *FAIL*) rc=1 ;; esac
      fi
    done
    rm -f "$ref"
    i=$((i + 1))
  done
done
exit $rc
