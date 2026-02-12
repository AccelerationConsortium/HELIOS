#!/bin/bash
# 简化自动录屏脚本（无需特殊权限）
# 使用ffmpeg录制全屏 + 自动触发demo

set -e

echo ""
echo "🎬 OTbot 自动录屏系统"
echo "================================"
echo ""

# 配置
API_BASE="http://localhost:8000"
DURATION=50  # 录制时长
OUTPUT_FILE="$HOME/Desktop/OTbot_Demo_$(date +%Y%m%d_%H%M%S).mp4"

# 1. 检查ffmpeg
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
    echo "❌ Backend未运行"
    echo "   请先运行: python3 -m uvicorn app.main:app --port 8000"
    exit 1
fi
echo "✅ Backend运行中"

# 3. 打开浏览器
echo ""
echo "🌐 打开浏览器..."
open "$API_BASE/static/lab.html"
sleep 3
echo "✅ 浏览器已打开"

# 4. 提示用户调整窗口
echo ""
echo "📐 请调整浏览器窗口:"
echo "   • 将浏览器窗口调整到合适大小"
echo "   • 确保前端UI完全可见"
echo "   • 隐藏不必要的工具栏/书签栏"
echo ""
read -p "准备好后按Enter开始录制... " -r
echo ""

# 5. 准备录屏
echo "📹 准备录屏..."
echo "   输出: $OUTPUT_FILE"
echo "   时长: ${DURATION}秒"
echo ""
echo "🎬 录屏将在3秒后开始..."
sleep 1 && echo "   3..."
sleep 1 && echo "   2..."
sleep 1 && echo "   1..."

# 6. 启动ffmpeg录屏（全屏）
echo ""
echo "🔴 录制中..."

# 后台启动ffmpeg
ffmpeg -f avfoundation \
    -i "1:0" \
    -t $DURATION \
    -r 30 \
    -c:v libx264 \
    -preset ultrafast \
    -pix_fmt yuv420p \
    -c:a aac \
    "$OUTPUT_FILE" \
    > /tmp/ffmpeg_record.log 2>&1 &

FFMPEG_PID=$!
sleep 2

# 7. 触发Demo
echo "🚀 触发demo campaign..."

CAMPAIGN_RESPONSE=$(curl -s -X POST "$API_BASE/api/v1/orchestrate/demo" \
    -H "Content-Type: application/json" \
    -d '{"objective_kpi": "overpotential_eta10", "max_rounds": 2}')

CAMPAIGN_ID=$(echo "$CAMPAIGN_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('campaign_id', 'unknown'))" 2>/dev/null || echo "unknown")

echo "✅ Demo started: $CAMPAIGN_ID"
echo ""
echo "💡 观察前端UI的实时更新:"
echo ""
echo "   📍 Round 1: LHS策略（~20秒）"
echo "      🤖 PlannerAgent → 选择策略"
echo "      📊 CandidateGenerator → 生成14D recipe"
echo "      🛡️  SafetyAgent → 5项安全检查"
echo "      🔧 CompilerAgent → 11步protocol"
echo "      ⚗️  Executor → 硬件执行"
echo "         • P300清洗reactor"
echo "         • P20分配10种precursor"
echo "         • 电化学沉积（10 mA/cm², 45s）"
echo "         • Camera拍照QC"
echo "         • Potentiostat HER测试"
echo "      🔍 SensingAgent → QC验证"
echo "      📈 η10 = 127.3 mV"
echo ""
echo "   📍 Round 2: Bayesian优化（~20秒）"
echo "      🎯 StrategySelector → 切换Bayesian"
echo "      🧠 CandidateGenerator → KNN + EI"
echo "      (重复执行...)"
echo "      📈 η10 = 89.7 mV ⬇️ (29.5% improvement!)"
echo ""

# 8. 监控进度
ELAPSED=0
while [ $ELAPSED -lt $DURATION ]; do
    if ! ps -p $FFMPEG_PID > /dev/null 2>&1; then
        echo ""
        echo "⚠️  录制已停止"
        break
    fi

    REMAINING=$((DURATION - ELAPSED))
    MINS=$((REMAINING / 60))
    SECS=$((REMAINING % 60))
    printf "\r⏱️  录制进度: %02d:%02d 剩余 " $MINS $SECS

    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

# 9. 等待完成
echo ""
echo ""
echo "⏹️  完成录制..."
wait $FFMPEG_PID 2>/dev/null || true
sleep 2

# 10. 检查结果
echo ""
echo "================================"
if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

    echo "✅ 录屏成功!"
    echo ""
    echo "📹 输出文件:"
    echo "   $OUTPUT_FILE"
    echo "   大小: $FILE_SIZE"
    echo ""
    echo "🎬 用途:"
    echo "   • Presentation演示"
    echo "   • 投资人展示"
    echo "   • 技术文档"
    echo "   • GitHub README"
    echo ""

    # 预览
    read -p "是否预览? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "$OUTPUT_FILE"
    fi
else
    echo "❌ 录屏失败"
    echo ""
    echo "查看日志: /tmp/ffmpeg_record.log"
    if [ -f /tmp/ffmpeg_record.log ]; then
        echo ""
        echo "最近错误:"
        tail -20 /tmp/ffmpeg_record.log
    fi
fi

echo "================================"
echo ""
