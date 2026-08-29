#!/usr/bin/env bash
# Keep GB10 page cache small during model load so NVRM can allocate the KV slab.
# Runs for 25 min max, flushes whenever Cached > 40 GiB. NVIDIA KB 5776 remedy.
end=$((SECONDS+1500))
while [ $SECONDS -lt $end ]; do
  c=$(awk '/^Cached:/{print int($2/1048576)}' /proc/meminfo)
  if [ "${c:-0}" -gt 40 ]; then
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  fi
  sleep 5
done
