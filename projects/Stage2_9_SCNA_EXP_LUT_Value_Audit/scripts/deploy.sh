#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
flavor="fair_combined"
remote_dir="${STAGE29_REMOTE_DIR:-/data/local/tmp/stage2_9_value_audit}"

usage() { echo "Usage: $0 [--flavor fair_combined|lut_only|scna_only] [--remote-dir PATH]" >&2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --flavor) [[ $# -ge 2 ]] || { usage; exit 2; }; flavor="$2"; shift 2 ;;
    --remote-dir) [[ $# -ge 2 ]] || { usage; exit 2; }; remote_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
case "$flavor" in fair_combined|lut_only|scna_only) ;; *) usage; exit 2 ;; esac

host_dir="$project_dir/artifacts/host"
dsp_dir="$project_dir/artifacts/$flavor"
for path in "$host_dir/htp_ops_test" "$host_dir/libhtp_ops.so" "$dsp_dir/libhtp_ops_skel.so"; do
  [[ -f "$path" ]] || { echo "Missing artifact: $path; run scripts/build_variants.sh" >&2; exit 1; }
done

adb get-state >/dev/null
adb shell "mkdir -p '$remote_dir/cdsp' '$remote_dir/dsp'"
adb push "$host_dir/htp_ops_test" "$host_dir/libhtp_ops.so" "$remote_dir/" >/dev/null
if [[ -f "$host_dir/libscna_env.so" ]]; then
  adb push "$host_dir/libscna_env.so" "$host_dir/scna_env_smoke" "$remote_dir/" >/dev/null
fi
adb push "$dsp_dir/libhtp_ops_skel.so" "$remote_dir/cdsp/" >/dev/null
adb push "$dsp_dir/libhtp_ops_skel.so" "$remote_dir/dsp/" >/dev/null
if [[ -f "$dsp_dir/libscna_env_skel.so" ]]; then
  adb push "$dsp_dir/libscna_env_skel.so" "$remote_dir/cdsp/" >/dev/null
  adb push "$dsp_dir/libscna_env_skel.so" "$remote_dir/dsp/" >/dev/null
fi
adb shell "chmod 755 '$remote_dir/htp_ops_test'; cd '$remote_dir' && sha256sum htp_ops_test libhtp_ops.so cdsp/libhtp_ops_skel.so" \
  >"$project_dir/manifests/device_${flavor}.sha256"
echo "$remote_dir"
