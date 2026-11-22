# Sourced (PC side) by tools/ops/day/snap.sh, podcheck_body.sh and stop_body.sh: a payload travels
# by file transfer (a detached `runpodctl send` here, `runpodctl receive` on the pod), never inside
# the exec-pod body, which an agent reproduces in a tool call and long base64 does not survive
# (RUNSH_SHA_MISMATCH three times in a row, 21 Nov). The body is one short script: receive under a
# timeout, check the file's sha256 against the PC's, then start. (Not in common.sh: common.sh is an
# input of the table key.)
# BODY_LIM: the most characters of a body's script; the tests hold every printed body under it.
BODY_LIM=1000
POD_RPC=/work/_bin/runpodctl
# send_start <dir> <file> <timeout s>: start `runpodctl send <file>` in <dir>, detached, under
# timeout -k 10 <s> (a body never sent ends the send: rc 124); <dir>/send.log gets the send's
# output and always ends SEND_RC=<n> (wait.sh reads it). Sets CODE (the one-time relay code word,
# only [a-z0-9-] kept) and SPID. No code within 30 s: prints send.log, stops the send, returns 4.
send_start() { local i
  (cd "$1" && HOME="$RENT/home" USERPROFILE="$RENT/home" exec nohup bash -c 'timeout -k 10 "$2" "$0" send "$1" & c=$!; trap "kill $c" TERM; wait "$c"; echo "SEND_RC=$?"' "$RPC" "$2" "$3" > send.log 2>&1 < /dev/null) &
  SPID=$!
  for i in $(seq 1 60); do grep -q "code is" "$1/send.log" 2>/dev/null && break; sleep 0.5; done
  CODE=$(grep -o "code is: [^ ]*" "$1/send.log" | head -1 | sed 's/code is: //' | tr -dc 'a-z0-9-' || true)
  [ -n "$CODE" ] || { echo "no code; send.log:"; cat "$1/send.log"; kill "$SPID" 2>/dev/null || true; return 4; }; }
# recv_frag <code> <file> <sha256> <timeout s>: the pod-side lines (sh, in the receiving folder,
# no single quote, so it fits inside sh -c '...'): receive <file> under timeout -k 10 <s>
# (output in receive.log; a failed or timed-out receive prints RECEIVE_FAILED rc=<n> and the
# log's last 3 lines, exit 1), then check its sha256 (else PAYLOAD_SHA_MISMATCH <sha>, the file
# removed, exit 1). Nothing after it runs on either failure.
recv_frag() {
  printf '%s' "timeout -k 10 $4 $POD_RPC receive $1 > receive.log 2>&1 || { echo RECEIVE_FAILED rc=\$?; tail -3 receive.log; exit 1; }; s=\$(sha256sum $2 | cut -c1-64); [ x\$s = x$3 ] || { echo PAYLOAD_SHA_MISMATCH \$s; rm -f $2; exit 1; }"; }
# tool_body <tool> <tar> <sha256> <code> <cmd>: the script of a pod-tool body (pod check, stop).
# Each job has its own folder d=/work/_podcheck/<tool>-<code> (no spaces: paths stay unquoted; the code word is used once, so two
# jobs on one pod never share one; an existing folder: FOLDER_EXISTS <d>, exit 1); sibling
# <tool>-* folders whose job ended (rc) over 10 minutes ago are removed. In the background:
# receive <tar> (recv_frag, 120 s), unpack it (else UNPACK_FAILED), <cmd>; its output goes to
# <d>/out, its exit code to <d>/rc. The body waits up to 20 s for rc, prints out and exits with rc;
# still running: prints out, then "PENDING <d>/out" (exit 0): exec cat of that file once more.
tool_body() {
  printf '%s' "t=/work/_podcheck/$1; d=\$t-$4; mkdir -p \${t%/*}; mkdir \$d && cd \$d || { echo FOLDER_EXISTS \$d; exit 1; }; find \${t%/*} -path \$t-\*/rc -mmin +10 | sed s,/rc\$,, | xargs rm -rf; nohup sh -c '($(recv_frag "$4" "$2" "$3" 120); tar --no-same-owner -xf $2 || { echo UNPACK_FAILED; exit 1; }; $5); echo \$? > rc.t && mv rc.t rc' > out 2>&1 < /dev/null & i=0; while [ ! -f rc ] && [ \$i -lt 20 ]; do sleep 1; i=\$((i + 1)); done; cat out; [ -f rc ] || { echo PENDING \$d/out; exit 0; }; exit \$(cat rc)"; }
# body_json <script>: the exec-pod body (one line): ["sh","-c",<script>], the script JSON-escaped.
# Prints REFUSED and returns 3 when the script is over BODY_LIM characters.
body_json() { local s="$1"
  [ "${#s}" -le "$BODY_LIM" ] || { echo "REFUSED: exec body script is ${#s} characters, over $BODY_LIM"; return 3; }
  s=${s//\\/\\\\}; s=${s//\"/\\\"}
  printf '{"command": ["sh", "-c", "%s"], "timeoutSeconds": 25}\n' "$s"; }
