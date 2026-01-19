#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <iterations>"
  exit 1
fi

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
  prompt="$(cat productize/PROMPT.md)"
  claude_cmd=(claude -p "$prompt" --output-format text --dangerously-skip-permissions)

  if command -v script >/dev/null 2>&1; then
    # `script` writes to a log file by default; using /dev/null means it prints to stdout.
    # We still tee to a temp file so we can read it back into $result after completion.
    script -q -e -c "$(printf '%q ' "${claude_cmd[@]}")" /dev/null 2>&1 | tee "$tmp_log" >&3
  else
    "${claude_cmd[@]}" 2>&1 | tee "$tmp_log" >&3
  fi

  result="$(cat "$tmp_log")"
  rm -f "$tmp_log"

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