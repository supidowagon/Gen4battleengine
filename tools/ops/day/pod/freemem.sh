#!/usr/bin/env bash
# POD SIDE. Prints whole GB free in the container: memory.max - (memory.current - inactive_file).
# (`free` shows the host's 1 TB, not this pod's 64 GB limit.)
MAX=$(cat /sys/fs/cgroup/memory.max); CUR=$(cat /sys/fs/cgroup/memory.current)
INACT=$(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat)
echo $(( (MAX - (CUR - INACT)) / 1073741824 ))
