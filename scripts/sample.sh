
#!/usr/bin/env bash
set -euo pipefail

concurrency="${CONCURRENCY:-4}"

seq 0 10000 | xargs -n1 -P "$concurrency" -I{} uv run test-hard.py \
  --world 1 \
  --stage 1 \
  --action_type jump \
  --log

uv run scripts/upload_hf.py --world 1 --stage 1 --repo_id junyeong-nero/super-mario-bros-1-1
