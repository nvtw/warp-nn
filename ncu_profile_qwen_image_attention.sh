#!/usr/bin/env bash
set -euo pipefail

repo=/home/twidmer/Documents/git/warp-nn
ncu=/usr/local/cuda-13.2/bin/ncu
python_bin="$repo/.venv/bin/python"
report=/tmp/warp_nn_qwen_image_attention
summary="${report}.txt"
probe=$(mktemp /tmp/warp_nn_qwen_image_attention.XXXXXX.py)
trap 'rm -f "$probe"' EXIT

for path in "$repo" "$ncu" "$python_bin"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required path: $path" >&2
        exit 1
    fi
done

cat >"$probe" <<'PY'
import warp as wp

from warp_nn.runtime.operators import BidirectionalGQAPlan

device = wp.get_device("cuda:0")
shape = (1, 24, 6905, 128)
query = wp.zeros(shape, dtype=wp.bfloat16, device=device)
key = wp.zeros(shape, dtype=wp.bfloat16, device=device)
value = wp.zeros(shape, dtype=wp.bfloat16, device=device)
BidirectionalGQAPlan(query, key, value).execute()
wp.synchronize_device(device)
PY

cd "$repo"
export PYTHONPATH="$repo"
"$ncu" \
    --force-overwrite \
    --target-processes all \
    --kernel-name 'regex:.*tiled_bidirectional_gqa_attention.*' \
    --launch-count 1 \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    --export "$report" \
    "$python_bin" "$probe"
"$ncu" --import "${report}.ncu-rep" --page details --print-details all >"$summary"
chmod 0644 "${report}.ncu-rep" "$summary"
echo "Nsight Compute report: ${report}.ncu-rep"
echo "Text summary: $summary"
