#!/usr/bin/env bash
# ============================================================
# Record Sensing Layer Demo with asciinema
# ============================================================
#
# This script records the sensing layer demos to .cast files.
#
# Usage:
#   ./demo/record_sensing_demo.sh                # Record all demos
#   ./demo/record_sensing_demo.sh --demo 1       # Record specific demo
#   ./demo/record_sensing_demo.sh --upload       # Record and upload
#   ./demo/record_sensing_demo.sh --convert      # Record + convert to GIF/MP4
#
# Output:
#   demo/recordings/sensing-demo-{1,2,3,4}.cast
#   demo/recordings/sensing-demo-all.cast
#
# ============================================================

set -e
cd "$(dirname "$0")/.."

RECORDING_DIR="demo/recordings"
UPLOAD=false
CONVERT=false
DEMO_NUM=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --demo)
            DEMO_NUM="$2"
            shift 2
            ;;
        --upload)
            UPLOAD=true
            shift
            ;;
        --convert)
            CONVERT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure recording directory exists
mkdir -p "$RECORDING_DIR"

# Check asciinema is installed
if ! command -v asciinema &> /dev/null; then
    echo "Error: asciinema is not installed."
    echo "Install with: brew install asciinema"
    exit 1
fi

# Demo titles
declare -A DEMO_TITLES
DEMO_TITLES[1]="Blind vs Sensing-Aware Recovery"
DEMO_TITLES[2]="Real-time Sensor Panel"
DEMO_TITLES[3]="Incident Replay Analysis"
DEMO_TITLES[4]="SafetyAdvisor Integration"
DEMO_TITLES[all]="Complete Sensing Layer Demo"

record_demo() {
    local demo_id="$1"
    local cast_file="$RECORDING_DIR/sensing-demo-${demo_id}.cast"
    local title="${DEMO_TITLES[$demo_id]}"

    echo ""
    echo "============================================================"
    echo "  Recording: $title"
    echo "  Output:    $cast_file"
    echo "============================================================"
    echo ""

    local cmd
    if [[ "$demo_id" == "all" ]]; then
        cmd="uv run python -m demo.run_sensing_demo"
    else
        cmd="uv run python -m demo.run_sensing_demo --demo $demo_id"
    fi

    asciinema rec \
        --overwrite \
        --title "Exp-Agent Sensing Layer: $title" \
        --idle-time-limit 2 \
        --command "$cmd" \
        "$cast_file"

    echo ""
    echo "  ✓ Saved: $cast_file"

    if $CONVERT; then
        convert_recording "$cast_file"
    fi

    if $UPLOAD; then
        echo "  Uploading to asciinema.org..."
        asciinema upload "$cast_file"
    fi
}

convert_recording() {
    local cast_file="$1"
    local base="${cast_file%.cast}"
    local gif_file="${base}.gif"
    local mp4_file="${base}.mp4"

    echo ""
    echo "  Converting to GIF and MP4..."

    # Check if agg is installed
    if command -v agg &> /dev/null; then
        echo "  → Creating GIF: $gif_file"
        agg --theme monokai \
            --font-size 14 \
            --cols 100 \
            --rows 40 \
            "$cast_file" \
            "$gif_file"
        echo "  ✓ GIF created: $gif_file"
    else
        echo "  ⚠ agg not installed. Install with: brew install agg"
    fi

    # Convert GIF to MP4 if ffmpeg is available
    if [[ -f "$gif_file" ]] && command -v ffmpeg &> /dev/null; then
        echo "  → Creating MP4: $mp4_file"
        ffmpeg -y -i "$gif_file" \
            -movflags faststart \
            -pix_fmt yuv420p \
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
            "$mp4_file" 2>/dev/null
        echo "  ✓ MP4 created: $mp4_file"
    elif [[ -f "$gif_file" ]]; then
        echo "  ⚠ ffmpeg not installed. Install with: brew install ffmpeg"
    fi
}

# Print header
echo ""
echo "============================================================"
echo "  Sensing Layer Demo Recording"
echo "============================================================"
echo ""
echo "  Terminal:  $(tput cols)x$(tput lines)"
echo "  Recommended: 100x40 or larger"
echo ""

# Record specific demo or all
if [[ -n "$DEMO_NUM" ]]; then
    record_demo "$DEMO_NUM"
else
    # Record each demo separately
    for i in 1 2 3 4; do
        record_demo "$i"
        echo ""
        echo "  Waiting 2 seconds before next recording..."
        sleep 2
    done

    # Also record the complete run
    echo ""
    echo "  Recording complete demo (all 4 scenarios)..."
    record_demo "all"
fi

echo ""
echo "============================================================"
echo "  Recording Complete!"
echo "============================================================"
echo ""
echo "  Recordings saved in: $RECORDING_DIR/"
echo ""
echo "  Playback:"
echo "    asciinema play $RECORDING_DIR/sensing-demo-1.cast"
echo "    asciinema play --speed 2 $RECORDING_DIR/sensing-demo-all.cast"
echo ""
echo "  Manual conversion:"
echo "    agg $RECORDING_DIR/sensing-demo-1.cast $RECORDING_DIR/sensing-demo-1.gif"
echo "    ffmpeg -i $RECORDING_DIR/sensing-demo-1.gif -pix_fmt yuv420p $RECORDING_DIR/sensing-demo-1.mp4"
echo ""
