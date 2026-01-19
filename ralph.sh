#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <iterations>"
  exit 1
fi

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

for ((i=1; i<=$1; i++)); do
  echo "Iteration $i"
  echo "--------------------------------"
  
  # Stream output live to stdout while also capturing it for the COMPLETE check.
  # NOTE: Many CLIs buffer output when not attached to a TTY. `script` forces a pseudo-TTY.
  tmp_log="$(mktemp)"
  tmp_runner="$(mktemp)"
  cat > "$tmp_runner" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
claude -p "$(cat productize/PROMPT.md)" --output-format text --dangerously-skip-permissions
SH
  chmod +x "$tmp_runner"

  if command -v script >/dev/null 2>&1; then
    # Run under a pseudo-TTY to force unbuffered streaming from the CLI.
    # We avoid `script -c "<shell string>"` quoting pitfalls by invoking a real executable script.
    script -q -e -c "$tmp_runner" /dev/null 2>&1 | tee "$tmp_log" >&3
  else
    "$tmp_runner" 2>&1 | tee "$tmp_log" >&3
  fi

  result="$(cat "$tmp_log")"
  rm -f "$tmp_log" "$tmp_runner"

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