#!/usr/bin/env bash
set -euo pipefail

# GLM-5.3-Flash-NVFP4 on Reddie (head, rank 0) + Spark4 (worker, rank 1), vLLM TP2 over the fabric.
# Official day-0 image (vLLM has glm5_next; SGLang support for this NVFP4 quant is still in-flight).
# Day-1: NO speculative decode. MTP phase-2 after base is stable (image has Glm5NextMTPModel).
# Run worker FIRST: Spark4 rank 1, wait ~20s, then Reddie rank 0.
NODE_RANK="${1:?usage: launch-glm53-vllm-tp2.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

IMAGE="ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8"
NAME="vllm_glm53"
# Checkpoint. The README's documented default is RedHatAI/GLM-5.3-Flash-NVFP4
# (compressed-tensors) because the ModelOpt builds emit intermittent corrupted token IDs
# -- vLLM #54150, measured 4/9/8 U+FFFD on a Hangul probe vs 0/0/0 for RedHatAI. The
# launchers previously hardcoded the ModelOpt path, so the shipped default did not match
# the documented one. Override with MODEL_HOST_PATH=... for the legacy/abliterated builds.
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/var/tmp/models/GLM-5.3-Flash-NVFP4-redhat}"
MODEL_PATH="/models/glm-5.3-flash-nvfp4"

# Guard: fail loudly if the resolved checkpoint is a ModelOpt build, unless the operator
# opted in. This is the mismatch that shipped for weeks -- the corruption is nearly
# invisible in English prose and only bites inside tool-call blocks, so it will not
# announce itself at boot.
if [ -f "$MODEL_HOST_PATH/config.json" ] && [ "${ALLOW_MODELOPT:-0}" != "1" ]; then
  _q=$(python3 -c "import json;print(json.load(open('$MODEL_HOST_PATH/config.json')).get('quantization_config',{}).get('quant_method',''))" 2>/dev/null || echo "")
  if [ "$_q" = "modelopt" ]; then
    echo "REFUSING: $MODEL_HOST_PATH is a ModelOpt build (quant_method=modelopt)." >&2
    echo "  ModelOpt NVFP4 emits intermittent corrupted token IDs (vLLM #54150)." >&2
    echo "  Use RedHatAI/GLM-5.3-Flash-NVFP4, or set ALLOW_MODELOPT=1 to override." >&2
    exit 5
  fi
fi

CACHE_HOST_PATH="/var/tmp/glm53-vllm-cache"
HEAD_IP="192.168.192.2"
MPORT="29521"
PORT="8000"

case "$NODE_RANK" in
  0) HOST_IP=192.168.192.2; HEADLESS="" ;;
  1) HOST_IP=192.168.192.4; HEADLESS="--headless" ;;
esac

test -f "$MODEL_HOST_PATH/config.json"
mkdir -p "$CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -e VLLM_HOST_IP=$HOST_IP \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=192.168.192.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 262144 \
    --max-num-seqs 6 --block-size 2304 --moe-backend marlin --speculative-config '{"method":"mtp","num_speculative_tokens":4}' --kv-cache-dtype fp8_e4m3 --kv-cache-memory 4445787956 \
    --enforce-eager --max-num-batched-tokens 8192 \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 --default-chat-template-kwargs '{"enable_thinking": false}' \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited; inspect with: docker logs $NAME" >&2
  exit 1
}
