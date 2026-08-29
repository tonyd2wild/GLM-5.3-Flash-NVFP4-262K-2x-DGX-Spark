#!/usr/bin/env bash
#
# GLM-5.3-Flash-DERISKED-NVFP4 + DFlash2 speculative decoding, TP2 on 2x DGX Spark.
# Upstream launch-glm53-vllm-tp2-dflash2.sh, retargeted to this fabric.
#
# Fleet:
#   rank 0 (head)   spark-1140  zeus@192.168.100.10  :8000
#   rank 1 (worker) gx10-05a3   zeus@192.168.100.11
#   fabric          192.168.100.0/24 on enp1s0f0np0 / rocep1s0f0 (QSFP port 1)
#
# Target weights: Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4
# Drafter:        incoai/GLM-5.3-Flash-DFlash2  (cannot be served alone)
# Image:          radixark/vllm-glm53-flash:sm121-v11-dflash2
#                 (retag of ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2)
#
# Prerequisites on BOTH nodes:
#   1. docker tag ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 \
#                 radixark/vllm-glm53-flash:sm121-v11-dflash2
#   2. weights at $MODEL_HOST_PATH
#   3. drafter at /var/tmp/models/GLM-5.3-Flash-DFlash2
#   4. $HOME/patches/sparse_attn_indexer_kpool.py  -- SM121 top-k fix.
#      Without it the engine dies on any decode past ~24K context.
#      (docs/SM121-CRASH-FORENSICS-2026-08-27.md)
#
# Usage: ./launch-glm53-vllm-tp2-dflash2.sh <0|1>   -- worker (1) FIRST, then head (0)
# Run cache_flusher.sh alongside on BOTH nodes (GB10 NVRM allocator, see repo docs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NODE_RANK="${1:?usage: launch-glm53-vllm-tp2-dflash2.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

IMAGE="radixark/vllm-glm53-flash:sm121-v11-dflash2"
NAME="vllm_glm53"
MODEL_HOST_PATH="/var/tmp/glm-5.3-flash-derisked-nvfp4"
MODEL_PATH="/models/glm-5.3-flash-derisked-nvfp4"
DRAFT_HOST_PATH="/var/tmp/models/GLM-5.3-Flash-DFlash2"
DRAFT_PATH="/models/dflash2-draft"
CACHE_HOST_PATH="/var/tmp/glm53-vllm-cache"
PATCH_HOST_PATH="$HOME/patches/sparse_attn_indexer_kpool.py"
CHAT_TEMPLATE_HOST="$SCRIPT_DIR/chat_template_mm.jinja"
HEAD_IP="192.168.100.10"
MPORT="29521"
PORT="8000"

case "$NODE_RANK" in
  0) HOST_IP=192.168.100.10; HEADLESS="" ;;          # spark-1140
  1) HOST_IP=192.168.100.11; HEADLESS="--headless" ;;  # gx10-05a3
esac

# Preflight: fail before docker run so a missing artifact is not a mystery boot.
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  cat >&2 <<EOF
ERROR: DFlash2 overlay image not present: $IMAGE
  Pull:  docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
         docker tag  ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 $IMAGE
EOF
  exit 3
}
test -f "$MODEL_HOST_PATH/config.json" || {
  cat >&2 <<EOF
ERROR: target weights not found at $MODEL_HOST_PATH
  Fetch (BOTH nodes):
    hf download Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4 --local-dir $MODEL_HOST_PATH
EOF
  exit 3
}
test -f "$DRAFT_HOST_PATH/config.json" || {
  cat >&2 <<EOF
ERROR: DFlash2 drafter checkpoint not found at $DRAFT_HOST_PATH
  Fetch (BOTH nodes):
    hf download incoai/GLM-5.3-Flash-DFlash2 --local-dir $DRAFT_HOST_PATH
EOF
  exit 3
}
test -f "$PATCH_HOST_PATH" || {
  cat >&2 <<EOF
ERROR: SM121 top-k patch missing: $PATCH_HOST_PATH
  On BOTH nodes:
    mkdir -p "\$HOME/patches"
    cp "$SCRIPT_DIR/docker/sparse_attn_indexer_kpool_sm121.py" "\$HOME/patches/sparse_attn_indexer_kpool.py"
EOF
  exit 3
}
test -f "$CHAT_TEMPLATE_HOST" || {
  echo "ERROR: chat template missing: $CHAT_TEMPLATE_HOST" >&2
  exit 3
}

mkdir -p "$CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v "$CHAT_TEMPLATE_HOST:/models/chat_template_mm.jinja:ro" \
  -v "$DRAFT_HOST_PATH:$DRAFT_PATH:ro" \
  -v "$PATCH_HOST_PATH:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro" \
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
  -e NCCL_IB_ADDR_RANGE=192.168.100.0/24 \
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
    --max-num-seqs 6 --block-size 2304 --moe-backend marlin \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_PATH\",\"num_speculative_tokens\":5}" \
    --kv-cache-dtype fp8_e4m3 --kv-cache-memory 6442450944 --max-num-batched-tokens 8192 \
    --enforce-eager \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 --default-chat-template-kwargs '{"enable_thinking":false}' \
    --chat-template /models/chat_template_mm.jinja \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP image=$IMAGE"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited; inspect with: docker logs $NAME" >&2
  exit 1
}
