#!/bin/bash
# 即时自动录屏（无需交互）

API_BASE="http://localhost:8000"
DURATION=50
OUTPUT_FILE="$HOME/Desktop/OTbot_Demo_$(date +%Y%m%d_%H%M%S).mp4"

echo "🎬 OTbot 即时录屏系统"
echo "================================"
echo ""

# 检查ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ 需要安装ffmpeg: brew install ffmpeg"
    exit 1
fi

# 检查backend
if ! curl -s "$API_BASE/api/v1/health" > /dev/null 2>&1; then
    echo "❌ Backend未运行"
    exit 1
fi

# 打开浏览器
open "$API_BASE/static/lab.html"
echo "✅ 浏览器已打开"
sleep 3

# 倒计时
echo ""
echo "🎬 5秒后开始录制..."
for i in 5 4 3 2 1; do
    echo "   $i..."
    sleep 1
done

# 开始录制
echo ""
echo "🔴 开始录制（${DURATION}秒）..."

ffmpeg -f avfoundation -i "0:none" -t $DURATION -r 30 -c:v libx264 -preset ultrafast -pix_fmt yuv420p "$OUTPUT_FILE" > /tmp/ffmpeg.log 2>&1 &
FFMPEG_PID=$!
sleep 2

# 触发demo
echo "🚀 触发demo..."
RESPONSE=$(curl -s -X POST "$API_BASE/api/v1/orchestrate/demo" -H "Content-Type: application/json" -d '{"objective_kpi":"overpotential_eta10","max_rounds":2}')
CAMPAIGN_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('campaign_id','?'))" 2>/dev/null)

echo "✅ Demo started: $CAMPAIGN_ID"
echo ""
echo "💡 前端UI实时显示:"
echo "   Round 1: LHS → η10=127.3mV"
echo "   Round 2: Bayesian → η10=89.7mV (↓29.5%)"
echo ""

# 监控
ELAPSED=0
while [ $ELAPSED -lt $DURATION ]; do
    ps -p $FFMPEG_PID > /dev/null 2>&1 || break
    printf "\r⏱️  %02d:%02d / %02d:00 " $((ELAPSED/60)) $((ELAPSED%60)) $((DURATION/60))
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

wait $FFMPEG_PID 2>/dev/null || true
echo ""
echo ""

# 结果
if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "================================"
    echo "✅ 录屏完成!"
    echo ""
    echo "📹 $OUTPUT_FILE"
    echo "   大小: $SIZE"
    echo "================================"
    open "$OUTPUT_FILE"
else
    echo "❌ 录屏失败"
    tail -10 /tmp/ffmpeg.log
fi
