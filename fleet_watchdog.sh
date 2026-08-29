#!/usr/bin/env bash
# fleet_watchdog.sh -- auto-recovery for the GLM-5.3 vLLM multi-node fleet.
# Runs on spark-1140 (head, rank 0). Probes /health; on N consecutive failures,
# tears down ALL containers, runs the GB10 memory ritual, relaunches
# worker-first (rank 1) then the head (rank 0), waits for ready.
#
# vLLM v1 CANNOT recover a dead engine core. Docker restart policies are unsafe
# here: headless workers exit 0 on head death (on-failure never fires) and the
# dead head often never exits at all. Full orchestrated relaunch is the only cure.
set -u

### ---- config -------------------------------------------------------------
HEALTH_URL="http://127.0.0.1:8000/health"   # NOT /v1/models: that returns 200
                                            # even with a dead engine. /health
                                            # returns 503 on EngineDeadError.
CHECK_INTERVAL=60          # seconds between probes
FAIL_THRESHOLD=3           # consecutive failures before recovery fires
CURL_TIMEOUT=15            # per-probe timeout
READY_TIMEOUT=3600         # matches VLLM_ENGINE_READY_TIMEOUT_S in launch script
CONTAINER="vllm_glm53"
# Same path on every node, byte-identical copies. Rank 0 is launched through
# run_on too, so this path must exist locally as well.
LAUNCH_SCRIPT="$HOME/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/launch-glm53-vllm-tp2-dflash2.sh"
SSH_KEY="$HOME/.ssh/id_ed25519_shared"
SSH_OPTS=(-i "$SSH_KEY" -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
LOCKFILE="$HOME/.fleet_watchdog.lock"
LOGFILE="$HOME/fleet_watchdog.log"
POST_TEARDOWN_SLEEP=10     # let master-port TIME_WAIT / NVRM settle
INTER_WORKER_SLEEP=5

# rank -> ssh target; empty string = local (head). Launch order is the
# ARRAY ORDER below: worker 1 first, head 0 last.
# Two-node fleet: spark-1140 (.10) + gx10-05a3 (.11) on 192.168.100.0/24.
RANK_ORDER=(1 0)
declare -A NODE=(
  [1]="zeus@192.168.100.11"
  [0]=""
)
### -------------------------------------------------------------------------

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOGFILE"; }

run_on() {  # run_on <rank> <command string>
  local rank="$1"; shift
  local target="${NODE[$rank]}"
  if [[ -z "$target" ]]; then
    bash -lc "$*" >> "$LOGFILE" 2>&1
  else
    ssh "${SSH_OPTS[@]}" "$target" "$*" >> "$LOGFILE" 2>&1
  fi
}

healthy() { curl -sf -m "$CURL_TIMEOUT" -o /dev/null "$HEALTH_URL"; }

mem_ritual() {  # GB10 NVRM allocator hygiene (launch script requires it)
  local rank="$1"
  run_on "$rank" 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || echo "WARN: drop_caches failed (sudo -n?)"; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null || echo "WARN: compact_memory failed"'
}

recover() {
  log "=== RECOVERY START: $FAIL_THRESHOLD consecutive health failures ==="

  # 1. Tear down EVERYTHING first. A worker must never start while the old
  #    head is dying: it joins the stale TCPStore rendezvous, exits 0
  #    when the old head finally dies, and wedges the new head's rendezvous.
  for r in "${RANK_ORDER[@]}"; do
    log "teardown: docker rm -f $CONTAINER on rank $r (${NODE[$r]:-local})"
    run_on "$r" "docker rm -f $CONTAINER 2>/dev/null || true"
  done
  sleep "$POST_TEARDOWN_SLEEP"

  # 2. Memory ritual on all nodes AFTER teardown, BEFORE relaunch.
  for r in "${RANK_ORDER[@]}"; do
    log "mem ritual on rank $r"
    mem_ritual "$r"
  done

  # 3. Relaunch: worker rank 1, then head rank 0.
  for r in "${RANK_ORDER[@]}"; do
    log "launch rank $r on ${NODE[$r]:-local}"
    if ! run_on "$r" "$LAUNCH_SCRIPT $r"; then
      log "ERROR: launch of rank $r reported failure; continuing (head may still rendezvous)"
    fi
    [[ "$r" != "0" ]] && sleep "$INTER_WORKER_SLEEP"
  done

  # 4. Wait for the engine to come up (TP2 load takes many minutes).
  log "waiting up to ${READY_TIMEOUT}s for $HEALTH_URL"
  local waited=0
  until healthy; do
    sleep 30; waited=$((waited + 30))
    if (( waited >= READY_TIMEOUT )); then
      log "ERROR: fleet did not become healthy within ${READY_TIMEOUT}s -- will retry via main loop"
      return 1
    fi
  done
  log "=== RECOVERY COMPLETE: healthy after ${waited}s ==="
  return 0
}

### ---- main ---------------------------------------------------------------
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "fleet_watchdog already running (lock: $LOCKFILE)" >&2
  exit 1
fi
log "watchdog started (pid $$, interval ${CHECK_INTERVAL}s, threshold $FAIL_THRESHOLD)"

fails=0
while true; do
  if healthy; then
    (( fails > 0 )) && log "health OK again after $fails failure(s)"
    fails=0
  else
    fails=$((fails + 1))
    log "health FAIL ($fails/$FAIL_THRESHOLD): $HEALTH_URL"
    if (( fails >= FAIL_THRESHOLD )); then
      recover || log "recovery attempt failed; probing continues"
      fails=0
    fi
  fi
  sleep "$CHECK_INTERVAL"
done
