#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <iterations>"
  exit 1
fi

# Safer pipelines: we want to notice failures from `script`/`claude` even when piped to `tee`.
set -o pipefail

# Ensure we run from the repo root (so relative paths work no matter where invoked).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Fail fast if the CLI isn't available (otherwise the loop will be confusing).
if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not found in PATH. Fix PATH or install it, then re-run."
  exit 127
fi

# Keep a handle to the original stdout so we can stream output live
# while still capturing it via command substitution.
exec 3>&1

# If you Ctrl-C, stop the whole harness cleanly instead of continuing broken iterations.
STOP_REQUESTED=0
trap 'STOP_REQUESTED=1' INT

for ((i=1; i<=$1; i++)); do
  echo "Iteration $i"
  echo "--------------------------------"
  
  # Stream output live to stdout while also capturing it for the COMPLETE check.
  # NOTE: Many CLIs buffer output when not attached to a TTY. `script` forces a pseudo-TTY.
  tmp_log="$(mktemp)" || { echo "ERROR: mktemp failed"; exit 1; }
  tmp_runner="$(mktemp)" || { echo "ERROR: mktemp failed"; exit 1; }
  cat > "$tmp_runner" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
claude -p "$(cat productize/PROMPT.md)" --output-format text --dangerously-skip-permissions
SH
  chmod +x "$tmp_runner"

  # Run Claude and stream output; capture exit status robustly even with piping to tee.
  cmd_status=0
  if command -v script >/dev/null 2>&1; then
    # Run under a pseudo-TTY to force unbuffered streaming from the CLI.
    # We avoid `script -c "<shell string>"` quoting pitfalls by invoking a real executable script.
    script -q -e -c "$tmp_runner" /dev/null 2>&1 | tee "$tmp_log" >&3
    cmd_status="${PIPESTATUS[0]}"
  else
    "$tmp_runner" 2>&1 | tee "$tmp_log" >&3
    cmd_status="${PIPESTATUS[0]}"
  fi

  result="$(cat "$tmp_log")"
  rm -f "$tmp_log" "$tmp_runner"

  # If user requested stop (Ctrl-C), exit cleanly now.
  if [ "${STOP_REQUESTED}" -eq 1 ] || [ "${cmd_status}" -eq 130 ]; then
    echo ""
    echo "Interrupted (Ctrl-C). Stopping."
    exec 3>&-
    exit 130
  fi

  # Helpful diagnostics for flaky CLI/API failures.
  if [ "${cmd_status}" -ne 0 ]; then
    if [[ "$result" == *"No messages returned"* ]]; then
      echo ""
      echo "WARN: claude returned no messages (likely transient network/auth/rate-limit). Re-run after fixing login/connectivity."
    else
      echo ""
      echo "WARN: claude exited with status ${cmd_status}. Continuing to next iteration."
    fi
  fi

  if [[ "$result" == *"<promise>COMPLETE</promise>"* ]]; then
    echo "All tasks complete after $i iterations."
    exec 3>&-
    exit 0
  fi
  
  echo ""
  echo "--- End of iteration $i ---"
  echo ""
done

exec 3>&-
echo "Reached max iterations ($1)"
exit 1