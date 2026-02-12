#!/usr/bin/env bash
# ============================================================
# Record Exp-Agent Demo with asciinema
# ============================================================
#
# This script records the demo to a .cast file using asciinema.
#
# Usage:
#   ./demo/record_demo.sh                    # Record demo
#   ./demo/record_demo.sh --upload           # Record and upload to asciinema.org
#
# Output:
#   demo/recordings/exp-agent-demo.cast      # Recorded session
#
# To convert to GIF:
#   pip3 install agg                          # or: brew install agg
#   agg demo/recordings/exp-agent-demo.cast demo/recordings/exp-agent-demo.gif
#
# To convert to MP4 (via Docker):
#   docker run --rm -v $PWD:/data asciinema/asciicast2gif \
#     /data/demo/recordings/exp-agent-demo.cast /data/demo/recordings/exp-agent-demo.gif
#
# To play back:
#   asciinema play demo/recordings/exp-agent-demo.cast
#   asciinema play --speed 2 demo/recordings/exp-agent-demo.cast
#
# ============================================================

set -e
cd "$(dirname "$0")/.."

RECORDING_DIR="demo/recordings"
CAST_FILE="$RECORDING_DIR/exp-agent-demo.cast"
UPLOAD=false

[[ "$1" == "--upload" ]] && UPLOAD=true

# Ensure recording directory exists
mkdir -p "$RECORDING_DIR"

# Check asciinema is installed
if ! command -v asciinema &> /dev/null; then
    echo "Error: asciinema is not installed."
    echo "Install with: brew install asciinema"
    exit 1
fi

echo "============================================================"
echo "  Exp-Agent Demo Recording"
echo "============================================================"
echo ""
echo "  Output:    $CAST_FILE"
echo "  Terminal:  $(tput cols)x$(tput lines)"
echo ""
echo "  The demo script will run automatically inside the recording."
echo "  Press Ctrl+C to stop recording early."
echo ""
echo "============================================================"
echo ""

# Set ideal terminal size for recording
# (asciinema captures the current terminal size)
echo "Recommended terminal size: 100x35"
echo "Current: $(tput cols)x$(tput lines)"
echo ""

# Record the demo
# --overwrite: overwrite existing file
# --title: title shown on asciinema.org
# --idle-time-limit: cap idle time in playback
asciinema rec \
    --overwrite \
    --title "Exp-Agent: Recovery-Aware Execution Agent Demo" \
    --idle-time-limit 3 \
    --command "bash demo/run_demo.sh" \
    "$CAST_FILE"

echo ""
echo "============================================================"
echo "  Recording saved to: $CAST_FILE"
echo "============================================================"
echo ""
echo "  Playback:"
echo "    asciinema play $CAST_FILE"
echo "    asciinema play --speed 2 $CAST_FILE"
echo ""
echo "  Convert to GIF:"
echo "    brew install agg"
echo "    agg $CAST_FILE ${CAST_FILE%.cast}.gif"
echo ""
echo "  Convert to SVG (animated):"
echo "    pip3 install asciinema-player"
echo "    # or embed in HTML with asciinema-player.js"
echo ""

if $UPLOAD; then
    echo "  Uploading to asciinema.org..."
    asciinema upload "$CAST_FILE"
fi
