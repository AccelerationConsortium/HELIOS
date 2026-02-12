#!/bin/bash
# 自动录屏Demo脚本
# 使用macOS内置的screenshot工具 + 自动化

set -e

echo "🎬 OTbot Auto Recording Script"
echo "================================"
echo ""

# 配置
API_BASE="http://localhost:8000"
DURATION=45  # 录制时长（秒）
OUTPUT_FILE="$HOME/Desktop/OTbot_Demo_$(date +%Y%m%d_%H%M%S).mov"

# 1. 检查backend是否运行
echo "📡 Checking backend..."
if ! curl -s "$API_BASE/api/v1/health" > /dev/null 2>&1; then
    echo "❌ Backend not running. Starting..."
    cd /Users/sissifeng/OTbot
    python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > /dev/null 2>&1 &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
    sleep 5
else
    echo "✅ Backend is running"
fi

# 2. 打开浏览器到前端UI
echo ""
echo "🌐 Opening browser..."
open -a "Google Chrome" "$API_BASE/static/lab.html" 2>/dev/null || \
open -a "Safari" "$API_BASE/static/lab.html" 2>/dev/null || \
open "$API_BASE/static/lab.html"

sleep 3

# 3. 获取Chrome窗口位置和大小（用于录制区域）
echo ""
echo "📹 Preparing screen recording..."

# 使用osascript获取窗口信息
WINDOW_INFO=$(osascript -e 'tell application "Google Chrome" to get bounds of front window' 2>/dev/null || \
              osascript -e 'tell application "Safari" to get bounds of front window' 2>/dev/null || \
              echo "100, 100, 1400, 900")

echo "   Window bounds: $WINDOW_INFO"

# 4. 提示用户准备
echo ""
echo "🎥 准备录屏..."
echo "   输出文件: $OUTPUT_FILE"
echo "   录制时长: ${DURATION}秒"
echo ""
echo "   录屏将在3秒后自动开始..."
echo ""

sleep 1 && echo "   3..." && \
sleep 1 && echo "   2..." && \
sleep 1 && echo "   1..." && \
echo "   🔴 Recording started!"

# 5. 启动录屏（使用screencapture的视频模式）
# 注意：macOS screencapture不直接支持视频录制
# 使用screenshot utility的AppleScript接口
osascript <<EOF &
tell application "System Events"
    -- 使用screenshot utility录制
    do shell script "screencapture -V $DURATION \"$OUTPUT_FILE\""
end tell
EOF

RECORD_PID=$!
echo "   Recording PID: $RECORD_PID"

sleep 2

# 6. 触发Demo
echo ""
echo "🚀 Triggering demo campaign..."

CAMPAIGN_RESPONSE=$(curl -s -X POST "$API_BASE/api/v1/orchestrate/demo" \
    -H "Content-Type: application/json" \
    -d '{"objective_kpi": "overpotential_eta10", "max_rounds": 2}')

CAMPAIGN_ID=$(echo "$CAMPAIGN_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('campaign_id', 'unknown'))" 2>/dev/null || echo "unknown")

echo "✅ Demo started: $CAMPAIGN_ID"
echo ""
echo "💡 前端UI正在实时显示:"
echo "   • Agent思考过程"
echo "   • 工具调用链"
echo "   • 硬件操控步骤"
echo "   • 协议步骤"
echo ""

# 7. 等待录制完成
ELAPSED=0
while [ $ELAPSED -lt $DURATION ]; do
    REMAINING=$((DURATION - ELAPSED))
    printf "\r⏱️  Recording: %02d:%02d remaining" $((REMAINING / 60)) $((REMAINING % 60))
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

echo ""
echo ""
echo "✅ Recording completed!"
echo ""

# 8. 等待文件保存
sleep 2

# 9. 检查输出文件
if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "📹 录屏文件:"
    echo "   路径: $OUTPUT_FILE"
    echo "   大小: $FILE_SIZE"
    echo ""
    echo "🎬 录屏完成! 可用于presentation演示。"
else
    echo "⚠️  录屏文件未找到。可能需要手动录屏。"
    echo ""
    echo "💡 手动录屏方法:"
    echo "   1. 按 Cmd + Shift + 5"
    echo "   2. 选择'录制所选部分'"
    echo "   3. 框选浏览器窗口"
    echo "   4. 点击'录制'"
    echo "   5. 运行: python3 trigger_demo_ui.py"
fi

echo ""
echo "================================"
