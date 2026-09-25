#!/system/bin/sh
# launch_worker.sh - Android launch script for the NPU interpolation worker.
#
# Invoked by the lsfg-vk layer when TOML `worker_sh` is set:
#   sh launch_worker.sh <model.onnx> <npu_interp> <sockfd> [perf_mode]
#
# Sets up the bionic/QNN environment, then execs the worker, which serves
# the R/V socket protocol (see worker/npu_interp.cpp header). stdout/stderr
# of the worker land in lsfg.log via the layer's redirect.
#
# perf_mode is TOML npu_perf_mode (burst|balanced|power-saver|...).
#
# TODO(device): adjust LSFG_ROOT to where you deploy
#   npu_interp + model + lib/ (bionic + QNN stack).
set -u

MODEL="${1:?usage: launch_worker.sh <model> <bin> <sockfd> [perf_mode]}"
BIN="${2:?usage: launch_worker.sh <model> <bin> <sockfd> [perf_mode]}"
SOCKFD="${3:?usage: launch_worker.sh <model> <bin> <sockfd> [perf_mode]}"
PERF_MODE="${4:-burst}"

# TODO(device): point at the deployed QNN/bionic support libs.
LSFG_ROOT="${LSFG_ROOT:-/data/local/tmp/lsfg}"

export LD_LIBRARY_PATH="${LSFG_ROOT}/lib/aarch64-android:${LSFG_ROOT}/lib/jni/arm64-v8a:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="${LSFG_ROOT}/lib/hexagon-v73/unsigned"

exec "$BIN" "$MODEL" "$SOCKFD" "$PERF_MODE"
