#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
studio_dir="$(dirname -- "$(dirname -- "$script_dir")")/yam/control/single_arm"
command=(uv run --no-sync --project "$studio_dir" python "$script_dir/lift_both.py")
if [[ $# -eq 0 ]]; then
    command+=(--execute)
else
    command+=("$@")
fi
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
