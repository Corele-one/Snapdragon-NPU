#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
dsp_arch="v79"

usage() {
  echo "Usage: $0 [--dsp-arch v73|v79|v81]  # default: v79" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dsp-arch)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      dsp_arch="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

# Qualcomm's setup script reads optional undefined variables.  Temporarily
# disable nounset while sourcing it, then restore this script's strict mode.
set +u
# shellcheck source=use_hexagon_sdk_6_6.sh
source "$script_dir/use_hexagon_sdk_6_6.sh"
set -u
command -v build_cmake >/dev/null || {
  echo "Hexagon SDK setup did not provide build_cmake." >&2
  exit 1
}

cd "$htp_dir"
build_cmake android
build_cmake hexagon "DSP_ARCH=$dsp_arch" \
  FIGURE8_ENABLE_PROFILE_TIMERS=ON \
  FIGURE8_ENABLE_LUT_EXP=OFF

echo "Built SCNA environment and FlashAttention baselines for $dsp_arch."
