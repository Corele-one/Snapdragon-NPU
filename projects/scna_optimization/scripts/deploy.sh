#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
bundle="${1:-}"
remote_dir="${2:-/data/local/tmp/scna_optimization_v79}"

case "$bundle" in
  reference|scna) ;;
  *) echo "Usage: $0 reference|scna [remote-dir]" >&2; exit 2 ;;
esac

artifact_dir="$project_dir/artifacts/$bundle"
for file in htp_ops_test libhtp_ops.so libhtp_ops_skel.so; do
  [[ -s "$artifact_dir/$file" ]] || { echo "Missing $artifact_dir/$file; run scripts/build.sh" >&2; exit 1; }
done

adb get-state >/dev/null
adb shell "mkdir -p '$remote_dir/cdsp' '$remote_dir/dsp'"
adb push "$artifact_dir/htp_ops_test" "$remote_dir/" >/dev/null
adb push "$artifact_dir/libhtp_ops.so" "$remote_dir/" >/dev/null
adb push "$artifact_dir/libhtp_ops_skel.so" "$remote_dir/cdsp/" >/dev/null
adb push "$artifact_dir/libhtp_ops_skel.so" "$remote_dir/dsp/" >/dev/null
if [[ "$bundle" == scna ]]; then
  adb push "$artifact_dir/scna_env_smoke" "$remote_dir/" >/dev/null
  adb push "$artifact_dir/libscna_env.so" "$remote_dir/" >/dev/null
  adb push "$artifact_dir/libscna_env_skel.so" "$remote_dir/cdsp/" >/dev/null
  adb push "$artifact_dir/libscna_env_skel.so" "$remote_dir/dsp/" >/dev/null
fi
adb shell "chmod 755 '$remote_dir/htp_ops_test' '$remote_dir/scna_env_smoke' 2>/dev/null || true"
echo "deployed=$bundle remote=$remote_dir"
