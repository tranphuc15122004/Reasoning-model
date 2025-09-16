#!/usr/bin/env bash
set -euo pipefail

# Low-VRAM friendly evaluator using Hugging Face + 8-bit loading (bitsandbytes).
# Run this script from the repository's evaluation directory:
#   cd evaluation && bash sh/run_eval_hf_8bit.sh

# Configurable env vars with sensible defaults
: "${MODEL:=Qwen/Qwen2.5-Math-1.5B-Instruct}"
: "${OUT_DIR:=Qwen/Qwen2.5-Math-1.5B-Instruct/math_eval}"
: "${PROMPT:=qwen25-math-cot}"
: "${SPLIT:=test}"
: "${TEMP:=0}"
: "${TOP_P:=1}"
: "${NSAMPLING:=1}"
: "${MAX_TOKENS:=768}"
: "${NUM_SHOTS:=0}"
: "${SAVE:=1}"
: "${OVERWRITE:=1}"

# Comma-separated list of datasets to evaluate. Adjust as needed.
: "${DATASETS:=gsm8k,math,svamp,aqua,asdiv,tabmwp,mawps,cmath,cn_middle_school,college_math,gaokao_math_cloze,gaokao_math_qa,gaokao2023en,gaokao2024_I,gaokao2024_II,gaokao2024_mix,minerva_math,olympiadbench,sat_math,carp_en,mmlu_stem,aime24,amc23}"

echo "[run_eval_hf_8bit] MODEL=${MODEL}"
echo "[run_eval_hf_8bit] DATASETS=${DATASETS}"
echo "[run_eval_hf_8bit] MAX_TOKENS=${MAX_TOKENS} PROMPT=${PROMPT}"

EXTRA_FLAGS=()
[ "${SAVE}" = "1" ] && EXTRA_FLAGS+=("--save_outputs")
[ "${OVERWRITE}" = "1" ] && EXTRA_FLAGS+=("--overwrite")

python3 -u math_eval.py \
  --model_name_or_path "${MODEL}" \
  --data_names "${DATASETS}" \
  --output_dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --prompt_type "${PROMPT}" \
  --temperature "${TEMP}" \
  --n_sampling "${NSAMPLING}" \
  --top_p "${TOP_P}" \
  --max_tokens_per_call "${MAX_TOKENS}" \
  --num_shots "${NUM_SHOTS}" \
  --load_in_8bit \
  "${EXTRA_FLAGS[@]}"

echo "[run_eval_hf_8bit] Done. Outputs under outputs/${OUT_DIR}. Metrics printed above."
