"""
使用QuickTime Player自动录屏Demo

通过AppleScript控制QuickTime Player进行屏幕录制
"""
import subprocess
import time
import json
import urllib.request
import os

API_BASE = "http://localhost:8000"
DURATION = 45  # seconds

def run_applescript(script):
    """运行AppleScript命令"""
    proc = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True
    )
    return proc.returncode == 0, proc.stdout.strip()

def start_quicktime_recording():
    """启动QuickTime Player录屏"""
    print("🎬 启动QuickTime Player录屏...")

    # AppleScript to start screen recording
    script = '''
    tell application "QuickTime Player"
        activate
        delay 1
        -- Create new screen recording
        new screen recording
        delay 2

        -- Start recording (需要手动点击录制按钮，或使用GUI scripting)
        tell application "System Events"
            tell process "QuickTime Player"
                -- 点击录制按钮
                click button 1 of window "Screen Recording"
                delay 1
                -- 点击屏幕任意位置开始录制整个屏幕
                -- 或者可以拖动选择区域
            end tell
        end tell
    end tell
    '''

    success, output = run_applescript(script)
    return success

def stop_quicktime_recording(output_path):
    """停止QuickTime Player录屏并保存"""
    print("\n⏹️  停止录屏...")

    script = f'''
    tell application "QuickTime Player"
        tell front document
            stop
            delay 1

            -- Save the recording
            set savePath to POSIX file "{output_path}"
            save in savePath
            delay 2
            close
        end tell
    end tell
    '''

    success, output = run_applescript(script)
    return success

def trigger_demo():
    """触发demo campaign"""
    print("🚀 触发demo campaign...")

    try:
        data = json.dumps({
            "objective_kpi": "overpotential_eta10",
            "max_rounds": 2
        }).encode('utf-8')

        req = urllib.request.Request(
            f"{API_BASE}/api/v1/orchestrate/demo",
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            campaign_id = result.get('campaign_id', 'unknown')
            print(f"✅ Demo started: {campaign_id}")
            return campaign_id
    except Exception as e:
        print(f"❌ Failed to trigger demo: {e}")
        return None

def main():
    """主流程"""
    print("\n" + "🎬 " * 25)
    print("  OTbot 自动录屏Demo - QuickTime Player版本")
    print("🎬 " * 25 + "\n")

    # 1. 检查backend
    print("📡 检查backend...")
    try:
        response = urllib.request.urlopen(f"{API_BASE}/api/v1/health", timeout=5)
        print("✅ Backend运行中")
    except:
        print("❌ Backend未运行")
        print("   请先启动: python3 -m uvicorn app.main:app --port 8000")
        return

    # 2. 打开浏览器
    print("\n🌐 打开浏览器...")
    subprocess.run(["open", f"{API_BASE}/static/lab.html"], check=False)
    time.sleep(3)

    # 3. 准备输出路径
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.expanduser(f"~/Desktop/OTbot_Demo_{timestamp}.mov")

    print(f"\n📹 录屏配置:")
    print(f"   时长: {DURATION}秒")
    print(f"   输出: {output_path}")

    # 4. 提示用户
    print("\n⚠️  重要提示:")
    print("   由于macOS安全限制，QuickTime录屏需要手动操作：")
    print("   1. QuickTime将自动打开")
    print("   2. 请手动点击红色'录制'按钮")
    print("   3. 然后点击屏幕任意位置（录制整个屏幕）")
    print("   4. 或者拖动选择录制区域（推荐：框选浏览器窗口）")
    print("")
    input("准备好后按Enter继续...")

    # 5. 尝试启动QuickTime（需要用户手动操作）
    success = start_quicktime_recording()

    if not success:
        print("\n⚠️  自动启动QuickTime失败，请手动操作：")
        print("   1. 打开QuickTime Player")
        print("   2. 文件 → 新建屏幕录制")
        print("   3. 点击录制按钮")
        print("   4. 选择录制区域")
        input("\n开始录制后按Enter触发demo...")
    else:
        print("✅ QuickTime已启动")
        time.sleep(3)

    # 6. 触发demo
    campaign_id = trigger_demo()

    if campaign_id:
        print("\n💡 前端UI正在实时显示:")
        print("   • Round 1: LHS策略（探索）")
        print("   • Round 2: Bayesian优化（利用）")
        print("   • 完整的agent调用链和硬件操控")

    # 7. 倒计时
    print(f"\n⏱️  录制进行中...")
    for remaining in range(DURATION, 0, -5):
        mins, secs = divmod(remaining, 60)
        print(f"   剩余时间: {mins:02d}:{secs:02d}")
        time.sleep(5 if remaining > 5 else remaining)

    # 8. 停止录制
    print("\n✅ Demo完成!")
    print("\n请手动停止QuickTime录制:")
    print("   1. 点击菜单栏的QuickTime图标")
    print("   2. 点击'停止录制'按钮")
    print("   3. 文件 → 存储 → 选择保存位置")
    print(f"   建议保存到: {output_path}")

    print("\n" + "="*80)
    print("✅ 录屏流程完成！")
    print("="*80)

if __name__ == "__main__":
    main()
