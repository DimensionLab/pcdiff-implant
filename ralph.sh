#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <iterations>"
  exit 1
fi

# Keep a handle to the original stdout so we can stream output live
# while still capturing it via command substitution.
exec 3>&1

for ((i=1; i<=$1; i++)); do
  echo "Iteration $i"
  echo "--------------------------------"
  
  # Stream output to stdout (fd 3) while capturing full output into $result
  result=$(claude -p "$(cat productize/PROMPT.md)" --output-format text --dangerously-skip-permissions 2>&1 | tee /dev/fd/3) || true

  echo "$result"

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