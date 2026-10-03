#!/bin/bash
# Serve the run dashboards. TensorBoard reads the event files that RunLogger
# writes next to each run, so there is no service to sign in to and nothing to
# lose when a hosted product changes its terms.
#
# On the server:
#     bash scripts/serve_tensorboard.sh
# On your machine, forward the port first:
#     ssh -p 30153 -L 6006:localhost:6006 houyi@frp-egg.com
# Then open http://localhost:6006
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/tensorboard --logdir runs --port 6006 --host 127.0.0.1 "$@"
