#!/usr/bin/env bash
# ============================================================
# Generate Sensing Demo Videos (GIF + MP4)
# ============================================================
#
# One-click script to generate all demo videos.
#
# Usage:
#   ./demo/generate_sensing_videos.sh
#
# Prerequisites:
#   brew install asciinema agg ffmpeg
#
# Output:
#   demo/recordings/sensing-demo-1.{cast,gif,mp4}
#   demo/recordings/sensing-demo-2.{cast,gif,mp4}
#   demo/recordings/sensing-demo-3.{cast,gif,mp4}
#   demo/recordings/sensing-demo-4.{cast,gif,mp4}
#   demo/recordings/sensing-demo-all.{cast,gif,mp4}
#
# ============================================================

set -e
cd "$(dirname "$0")/.."

RECORDING_DIR="demo/recordings"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║  Sensing Layer Demo Video Generator                          ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check dependencies
check_dep() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}✗${NC} $1 not found. Install with: $2"
        return 1
    else
        echo -e "${GREEN}✓${NC} $1 found"
        return 0
    fi
}

echo -e "${BOLD}Checking dependencies...${NC}"
DEPS_OK=true
check_dep asciinema "brew install asciinema" || DEPS_OK=false
check_dep agg "brew install agg" || DEPS_OK=false
check_dep ffmpeg "brew install ffmpeg" || DEPS_OK=false
check_dep uv "curl -LsSf https://astral.sh/uv/install.sh | sh" || DEPS_OK=false

if ! $DEPS_OK; then
    echo ""
    echo -e "${RED}Please install missing dependencies and try again.${NC}"
    exit 1
fi

echo ""
echo -e "${BOLD}All dependencies found!${NC}"
echo ""

mkdir -p "$RECORDING_DIR"

# Demo info
declare -a DEMO_NAMES
DEMO_NAMES[1]="Blind vs Sensing-Aware"
DEMO_NAMES[2]="Real-time Sensor Panel"
DEMO_NAMES[3]="Incident Replay"
DEMO_NAMES[4]="SafetyAdvisor"

generate_demo() {
    local num="$1"
    local name="${DEMO_NAMES[$num]}"
    local base="$RECORDING_DIR/sensing-demo-$num"

    echo -e "${BOLD}${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${BOLD}Demo $num: $name${NC}"
    echo -e "${BOLD}${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Step 1: Record with asciinema
    echo -e "  ${CYAN}[1/3]${NC} Recording .cast file..."
    asciinema rec \
        --overwrite \
        --title "Sensing Demo $num: $name" \
        --idle-time-limit 2 \
        --command "uv run python -m demo.run_sensing_demo --demo $num --fast" \
        "${base}.cast" 2>/dev/null

    echo -e "        ${GREEN}✓${NC} ${base}.cast"

    # Step 2: Convert to GIF
    echo -e "  ${CYAN}[2/3]${NC} Converting to GIF..."
    agg --theme monokai \
        --font-size 14 \
        --cols 100 \
        --rows 40 \
        "${base}.cast" \
        "${base}.gif" 2>/dev/null

    echo -e "        ${GREEN}✓${NC} ${base}.gif"

    # Step 3: Convert to MP4
    echo -e "  ${CYAN}[3/3]${NC} Converting to MP4..."
    ffmpeg -y -i "${base}.gif" \
        -movflags faststart \
        -pix_fmt yuv420p \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        "${base}.mp4" 2>/dev/null

    echo -e "        ${GREEN}✓${NC} ${base}.mp4"
    echo ""
}

# Generate each demo
for i in 1 2 3 4; do
    generate_demo $i
done

# Generate all-in-one
echo -e "${BOLD}${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${BOLD}Complete Demo (All 4 Scenarios)${NC}"
echo -e "${BOLD}${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

base="$RECORDING_DIR/sensing-demo-all"

echo -e "  ${CYAN}[1/3]${NC} Recording .cast file..."
asciinema rec \
    --overwrite \
    --title "Sensing Layer: Complete Demo" \
    --idle-time-limit 2 \
    --command "uv run python -m demo.run_sensing_demo --fast" \
    "${base}.cast" 2>/dev/null
echo -e "        ${GREEN}✓${NC} ${base}.cast"

echo -e "  ${CYAN}[2/3]${NC} Converting to GIF..."
agg --theme monokai \
    --font-size 14 \
    --cols 100 \
    --rows 40 \
    "${base}.cast" \
    "${base}.gif" 2>/dev/null
echo -e "        ${GREEN}✓${NC} ${base}.gif"

echo -e "  ${CYAN}[3/3]${NC} Converting to MP4..."
ffmpeg -y -i "${base}.gif" \
    -movflags faststart \
    -pix_fmt yuv420p \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    "${base}.mp4" 2>/dev/null
echo -e "        ${GREEN}✓${NC} ${base}.mp4"
echo ""

# Summary
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║  Generation Complete!                                         ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Generated files:${NC}"
echo ""

for i in 1 2 3 4 all; do
    if [[ "$i" == "all" ]]; then
        name="Complete Demo"
    else
        name="${DEMO_NAMES[$i]}"
    fi
    base="$RECORDING_DIR/sensing-demo-$i"

    cast_size=$(du -h "${base}.cast" 2>/dev/null | cut -f1 || echo "?")
    gif_size=$(du -h "${base}.gif" 2>/dev/null | cut -f1 || echo "?")
    mp4_size=$(du -h "${base}.mp4" 2>/dev/null | cut -f1 || echo "?")

    if [[ "$i" == "all" ]]; then
        echo -e "  ${BOLD}📦 $name${NC}"
    else
        echo -e "  ${BOLD}Demo $i: $name${NC}"
    fi
    echo -e "     .cast: ${cast_size}  |  .gif: ${gif_size}  |  .mp4: ${mp4_size}"
done

echo ""
echo -e "${BOLD}Playback:${NC}"
echo "  asciinema play $RECORDING_DIR/sensing-demo-1.cast"
echo "  asciinema play --speed 2 $RECORDING_DIR/sensing-demo-all.cast"
echo ""
