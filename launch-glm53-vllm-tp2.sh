#!/usr/bin/env bash
set -euo pipefail

# GLM-5.3-Flash-DERISKED-NVFP4 MTP fallback on spark-1140 (head, rank 0) +
# gx10-05a3 (worker, rank 1), vLLM TP2 over 192.168.100.0/24.
# DFlash2 production lane is launch-glm53-vllm-tp2-dflash2.sh; this script is
# the MTP-4 fallback (sm121-v8, no drafter overlay).
# Run worker FIRST: gx10-05a3 rank 1, wait ~20s, then spark-1140 rank 0.
NODE_RANK="${1:?usage: launch-glm53-vllm-tp2.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

IMAGE="ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8"
NAME="vllm_glm53"
MODEL_HOST_PATH="/var/tmp/glm-5.3-flash-derisked-nvfp4"
MODEL_PATH="/models/glm-5.3-flash-derisked-nvfp4"
CACHE_HOST_PATH="/var/tmp/glm53-vllm-cache"
HEAD_IP="192.168.100.10"
MPORT="29521"
PORT="8000"

case "$NODE_RANK" in
  0) HOST_IP=192.168.100.10; HEADLESS="" ;;          # spark-1140
  1) HOST_IP=192.168.100.11; HEADLESS="--headless" ;;  # gx10-05a3
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
