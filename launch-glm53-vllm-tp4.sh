#!/usr/bin/env bash
set -euo pipefail

# GLM-5.3-Flash-NVFP4 TP4. This fleet has only TWO nodes (spark-1140 .10,
# gx10-05a3 .11) on 192.168.100.0/24. Ranks 2 and 3 have no hosts — use
# launch-glm53-vllm-tp2-dflash2.sh. Fill in rank 2/3 IPs if two more Sparks
# are cabled on. TP4 / 1M-context README numbers were measured on a different
# four-node 192.168.192.0/24 fabric.
NODE_RANK="${1:?usage: launch-glm53-vllm-tp4.sh <0|1|2|3>}"

IMAGE="ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8"
NAME="vllm_glm53"
MODEL_HOST_PATH="/var/tmp/glm-5.3-flash-derisked-nvfp4"
MODEL_PATH="/models/glm-5.3-flash-derisked-nvfp4"
CACHE_HOST_PATH="/var/tmp/glm53-vllm-cache"
HEAD_IP="192.168.100.10"
MPORT="29521"
PORT="8000"

case "$NODE_RANK" in
  0) HOST_IP=192.168.100.10; HEADLESS="" ;;            # spark-1140
  1) HOST_IP=192.168.100.11; HEADLESS="--headless" ;;  # gx10-05a3
  2) echo "rank 2: no third node on this fabric -- see header" >&2; exit 2 ;;
  3) echo "rank 3: no fourth node on this fabric -- see header" >&2; exit 2 ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac

test -f "$MODEL_HOST_PATH/config.json"
mkdir -p "$CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --memory 112g --memory-swap 112g \
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
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 1048576 \
    --max-num-seqs 6 --block-size 2304 --moe-backend marlin --speculative-config '{"method":"mtp","num_speculative_tokens":4}' --kv-cache-dtype fp8_e4m3 --kv-cache-memory 25769803776 \
    --enforce-eager \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 --chat-template /models/glm-5.3-flash-nvfp4/chat_template_mm.jinja --default-chat-template-kwargs '{"enable_thinking": false}' \
    --distributed-executor-backend mp \
    --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP tp4"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited; inspect with: docker logs $NAME" >&2
  exit 1
}
