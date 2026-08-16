#!/usr/bin/env bash
# Query per-shard progress of a running pass@k job.
#
#   bash scripts/qwen/passk_progress.sh            # one snapshot
#   watch -n 30 bash scripts/qwen/passk_progress.sh
#
# The runner prints "[qwen-async-passk] generated N/TOTAL ..." every 50
# completions per shard, so the last such line is the shard's position. Shards
# are independent engines, so they finish at different times; the ETA is driven
# by the slowest one.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-qwen3p8_27b_passk16_temp1p2_tp1_shards8_c16}"
LOG_DIR="${LOG_DIR:-logs/${RUN_TAG}}"

if [[ ! -d "${LOG_DIR}" ]]; then
  echo "no such log dir: ${LOG_DIR}"
  echo "set RUN_TAG or LOG_DIR; available:"
  ls -1d logs/*passk* 2>/dev/null || echo "  (none)"
  exit 1
fi

BASE_OUT="$(cat "${LOG_DIR}/.base_out" 2>/dev/null || echo "?")"
echo "run   : ${RUN_TAG}"
echo "out   : ${BASE_OUT}"
echo "time  : $(date -Is)"
echo

printf "%-7s %-10s %10s %10s %7s  %s\n" shard state done total pct last-update
printf "%-7s %-10s %10s %10s %7s  %s\n" ------ ---------- ---------- ---------- ------- -----------
sum_done=0; sum_total=0; running=0; finished=0
for f in "${LOG_DIR}"/shard*.log; do
  [[ -e "${f}" ]] || continue
  n="$(basename "${f}" .log)"
  line="$(grep -a 'generated [0-9]*/[0-9]*' "${f}" 2>/dev/null | tail -1)"
  done_n="$(sed -n 's/.*generated \([0-9]*\)\/\([0-9]*\).*/\1/p' <<<"${line}")"
  total_n="$(sed -n 's/.*generated \([0-9]*\)\/\([0-9]*\).*/\2/p' <<<"${line}")"
  done_n="${done_n:-0}"; total_n="${total_n:-0}"

  if grep -qa 'passk_summary.md' "${f}" 2>/dev/null; then
    state=done; finished=$((finished+1))
  elif grep -qa 'Traceback\|CUDA out of memory\|Error' "${f}" 2>/dev/null && ! kill -0 "$(pgrep -f "shard_index ${n#shard}" | head -1)" 2>/dev/null; then
    state=ERROR
  elif [[ "${done_n}" -gt 0 ]]; then
    state=running; running=$((running+1))
  else
    state=loading
  fi

  pct="-"
  [[ "${total_n}" -gt 0 ]] && pct="$(awk -v d="${done_n}" -v t="${total_n}" 'BEGIN{printf "%.1f%%", 100*d/t}')"
  mtime="$(date -d "@$(stat -c %Y "${f}")" '+%H:%M:%S' 2>/dev/null || echo '?')"
  printf "%-7s %-10s %10s %10s %7s  %s\n" "${n}" "${state}" "${done_n}" "${total_n}" "${pct}" "${mtime}"
  sum_done=$((sum_done + done_n)); sum_total=$((sum_total + total_n))
done

echo
if [[ "${sum_total}" -gt 0 ]]; then
  awk -v d="${sum_done}" -v t="${sum_total}" -v r="${running}" -v f="${finished}" \
    'BEGIN{printf "overall: %d/%d generations (%.1f%%)   shards running=%d done=%d\n", d, t, 100*d/t, r, f}'
else
  echo "overall: engines still loading (no generations yet)"
fi

echo
echo "GPU:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader | sed 's/^/  /'

echo
echo "errors seen in shard logs:"
hits="$(grep -al 'Traceback\|CUDA out of memory' "${LOG_DIR}"/shard*.log 2>/dev/null || true)"
[[ -n "${hits}" ]] && sed 's/^/  /' <<<"${hits}" || echo "  none"
n_gen_err="$(grep -hac 'generation_error' "${LOG_DIR}"/shard*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)"
echo "  generation_error lines: ${n_gen_err:-0}"

# This is a query tool: a clean report must exit 0, and the greps above return
# non-zero simply for finding nothing wrong.
exit 0
