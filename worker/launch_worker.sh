#!/system/bin/sh
# launch_worker.sh - Android launch script for the NPU interpolation worker.
#
# Invoked by the lsfg-vk layer when TOML `worker_sh` is set:
#   sh launch_worker.sh <model.onnx> <npu_interp> <sockfd>
#
# Sets up the bionic/QNN environment, then execs the worker, which serves
# the R/V socket protocol (see worker/npu_interp.cpp header). stdout/stderr
# of the worker land in lsfg.log via the layer's redirect.
#
# TODO(device): adjust LSFG_ROOT to where you deploy
#   npu_interp + model + lib/ (bionic + QNN stack).
set -u

MODEL="${1:?usage: launch_worker.sh <model> <bin> <sockfd>}"
BIN="${2:?usage: launch_worker.sh <model> <bin> <sockfd>}"
SOCKFD="${3:?usage: launch_worker.sh <model> <bin> <sockfd>}"

# TODO(device): point at the deployed QNN/bionic support libs.
LSFG_ROOT="${LSFG_ROOT:-/data/local/tmp/lsfg}"

export LD_LIBRARY_PATH="${LSFG_ROOT}/lib/aarch64-android:${LSFG_ROOT}/lib/jni/arm64-v8a:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="${LSFG_ROOT}/lib/hexagon-v73/unsigned"

exec "$BIN" "$MODEL" "$SOCKFD"
