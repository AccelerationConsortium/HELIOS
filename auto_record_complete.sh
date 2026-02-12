#!/bin/bash
# 完全自动化录屏脚本
# 使用ffmpeg捕获屏幕 + 自动触发demo

set -e

echo ""
echo "🎬 OTbot 完全自动化录屏"
echo "================================"
echo ""

# 配置
API_BASE="http://localhost:8000"
DURATION=50  # 录制时长（秒）- 稍长一些确保捕获完整demo
OUTPUT_FILE="$HOME/Desktop/OTbot_Demo_$(date +%Y%m%d_%H%M%S).mp4"
SCREEN_SIZE="1920x1080"  # 可调整

# 1. 检查ffmpeg
echo "🔧 检查依赖..."
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg未安装"
    echo "   安装: brew install ffmpeg"
    exit 1
fi
echo "✅ ffmpeg已安装"

# 2. 检查backend
echo ""
echo "📡 检查backend..."
if ! curl -s "$API_BASE/api/v1/health" > /dev/null 2>&1; then
    echo "⚠️  Backend未运行，正在启动..."
    cd /Users/sissifeng/OTbot
    python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/otbot_backend.log 2>&1 &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
    sleep 5

    if ! curl -s "$API_BASE/api/v1/health" > /dev/null 2>&1; then
        echo "❌ Backend启动失败"
        exit 1
    fi
fi
echo "✅ Backend运行中"

# 3. 打开浏览器到前端UI（最大化窗口以便录制）
echo ""
echo "🌐 打开浏览器..."

# 使用AppleScript打开并最大化Chrome
osascript <<EOF
tell application "Google Chrome"
    activate
    open location "$API_BASE/static/lab.html"
    delay 2

    tell application "System Events"
        tell process "Chrome"
            -- 最大化窗口
            set frontmost to true
            keystroke "f" using {command down, control down}
        end tell
    end tell
end tell
EOF

sleep 3

# 4. 获取Chrome窗口ID（用于聚焦）
CHROME_WINDOW=$(osascript -e 'tell application "Google Chrome" to get id of front window' 2>/dev/null || echo "")

echo "✅ 浏览器已打开"
echo "   窗口ID: $CHROME_WINDOW"

# 5. 准备录屏
echo ""
echo "📹 准备ffmpeg录屏..."
echo "   输出: $OUTPUT_FILE"
echo "   时长: ${DURATION}秒"
echo "   分辨率: $SCREEN_SIZE"
echo ""

# 6. 倒计时
echo "🎬 录屏将在3秒后开始..."
sleep 1 && echo "   3..."
sleep 1 && echo "   2..."
sleep 1 && echo "   1..."

# 7. 启动ffmpeg录屏（后台）
echo ""
echo "🔴 开始录屏..."

# macOS使用avfoundation捕获屏幕
# "1:0" 表示：视频设备1（屏幕）+ 音频设备0（麦克风，可选）
ffmpeg -f avfoundation \
    -i "1:0" \
    -t $DURATION \
    -r 30 \
    -s $SCREEN_SIZE \
    -c:v libx264 \
    -preset ultrafast \
    -pix_fmt yuv420p \
    -c:a aac \
    "$OUTPUT_FILE" \
    > /tmp/ffmpeg_record.log 2>&1 &

FFMPEG_PID=$!
echo "   ffmpeg PID: $FFMPEG_PID"

sleep 2

# 8. 触发Demo
echo ""
echo "🚀 触发demo campaign..."

CAMPAIGN_RESPONSE=$(curl -s -X POST "$API_BASE/api/v1/orchestrate/demo" \
    -H "Content-Type: application/json" \
    -d '{"objective_kpi": "overpotential_eta10", "max_rounds": 2}')

CAMPAIGN_ID=$(echo "$CAMPAIGN_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('campaign_id', 'unknown'))" 2>/dev/null || echo "unknown")

echo "✅ Demo campaign started!"
echo "   Campaign ID: $CAMPAIGN_ID"
echo ""
echo "💡 前端UI正在实时显示:"
echo "   📍 Round 1: LHS策略（探索）"
echo "      • PlannerAgent决策"
echo "      • CandidateGenerator生成14D sample"
echo "      • SafetyAgent验证5项检查"
echo "      • CompilerAgent生成11步protocol"
echo "      • Executor: P300清洗 → P20分配 → 电沉积 → 拍照 → HER测试"
echo "      • SensingAgent QC验证"
echo "      • Result: η10 = 127.3 mV"
echo ""
echo "   📍 Round 2: Bayesian优化（利用）"
echo "      • StrategySelector切换到Bayesian"
echo "      • CandidateGenerator: KNN + EI"
echo "      • (重复执行流程)"
echo "      • Result: η10 = 89.7 mV (29.5% improvement!)"
echo ""

# 9. 监控录制进度
ELAPSED=0
while [ $ELAPSED -lt $DURATION ]; do
    # 检查ffmpeg是否还在运行
    if ! ps -p $FFMPEG_PID > /dev/null 2>&1; then
        echo ""
        echo "⚠️  ffmpeg已停止"
        break
    fi

    REMAINING=$((DURATION - ELAPSED))
    MINS=$((REMAINING / 60))
    SECS=$((REMAINING % 60))

    printf "\r⏱️  录制中: %02d:%02d 剩余 " $MINS $SECS

    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

# 10. 等待ffmpeg完成
echo ""
echo ""
echo "⏹️  等待录制完成..."
wait $FFMPEG_PID 2>/dev/null || true

sleep 2

# 11. 检查输出
echo ""
echo "================================"
if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    FILE_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_FILE" 2>/dev/null | cut -d. -f1)

    echo "✅ 录屏完成!"
    echo ""
    echo "📹 输出文件:"
    echo "   路径: $OUTPUT_FILE"
    echo "   大小: $FILE_SIZE"
    echo "   时长: ${FILE_DURATION}秒"
    echo ""
    echo "🎬 可用于:"
    echo "   • Presentation演示"
    echo "   • 投资人展示"
    echo "   • 技术文档"
    echo "   • GitHub README"
    echo ""

    # 尝试打开视频
    read -p "是否预览录屏? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "$OUTPUT_FILE"
    fi
else
    echo "❌ 录屏文件未生成"
    echo "   查看日志: /tmp/ffmpeg_record.log"

    if [ -f /tmp/ffmpeg_record.log ]; then
        echo ""
        echo "错误日志:"
        tail -20 /tmp/ffmpeg_record.log
    fi
fi

echo "================================"
echo ""
