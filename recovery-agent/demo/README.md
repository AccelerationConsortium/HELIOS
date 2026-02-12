# Demo Recording Guide

## Quick Start

### Option 1: Record with asciinema (推荐)

在你自己的终端中运行，会有打字效果和完整的色彩显示：

```bash
# 1. 先调整终端窗口大小（推荐 100x35 或更大）

# 2. 录制
./demo/record_demo.sh

# 3. 回放
asciinema play demo/recordings/exp-agent-demo.cast
asciinema play --speed 2 demo/recordings/exp-agent-demo.cast   # 2倍速

# 4. 转换为 GIF
agg demo/recordings/exp-agent-demo.cast demo/recordings/exp-agent-demo.gif

# 5. 自定义 GIF（更好的视觉效果）
agg --theme monokai \
    --font-size 14 \
    --cols 100 \
    --rows 35 \
    demo/recordings/exp-agent-demo.cast \
    demo/recordings/exp-agent-demo-styled.gif
```

### Option 2: 手动录屏 + 运行脚本

如果你用 OBS / QuickTime / 屏幕录制工具：

```bash
# 在终端里直接运行 demo 脚本（有打字效果）
./demo/run_demo.sh

# 快速模式（无打字延迟）
./demo/run_demo.sh --fast
```

### Option 3: 上传到 asciinema.org

```bash
# 录制并上传（会给你一个分享链接）
./demo/record_demo.sh --upload
```

## 转换为视频格式

### GIF（推荐用于嵌入文档/PPT）

```bash
# 基础转换
agg demo/recordings/exp-agent-demo.cast demo/recordings/exp-agent-demo.gif

# 高质量版本
agg --theme monokai \
    --font-size 16 \
    --speed 1.5 \
    demo/recordings/exp-agent-demo.cast \
    demo/recordings/exp-agent-demo-hq.gif
```

### MP4（推荐用于演示）

方法一：使用 ffmpeg 从 GIF 转换

```bash
brew install ffmpeg
ffmpeg -i demo/recordings/exp-agent-demo.gif \
       -movflags faststart \
       -pix_fmt yuv420p \
       -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
       demo/recordings/exp-agent-demo.mp4
```

方法二：使用 QuickTime 录屏

1. 打开 QuickTime Player → File → New Screen Recording
2. 选择录制区域为终端窗口
3. 运行 `./demo/run_demo.sh`
4. 停止录制 → 导出为 MP4

### SVG 动画（推荐用于网页嵌入）

```bash
pip3 install svg-term
svg-term --cast demo/recordings/exp-agent-demo.cast \
         --out demo/recordings/exp-agent-demo.svg \
         --window
```

## 嵌入到网页

使用 asciinema player（支持 .cast 文件直接播放）：

```html
<div id="demo"></div>
<link rel="stylesheet" type="text/css" href="https://unpkg.com/asciinema-player@3.7/dist/bundle/asciinema-player.css" />
<script src="https://unpkg.com/asciinema-player@3.7/dist/bundle/asciinema-player.min.js"></script>
<script>
  AsciinemaPlayer.create('exp-agent-demo.cast', document.getElementById('demo'), {
    theme: 'monokai',
    speed: 1.5,
    idleTimeLimit: 3
  });
</script>
```

## Demo 内容

脚本展示 6 个场景：

| # | 场景 | 故障模式 | Agent 决策 |
|---|------|---------|-----------|
| 1 | 正常运行 | none | 正常执行 → 安全关机 |
| 2 | 温度过冲 | overshoot | 检测 → 分类为 UNSAFE → ABORT → 安全关机 |
| 3 | 通信超时 | timeout | 检测 → RETRY → 等待 2s → 重试成功 |
| 4 | 传感器故障 | sensor_fail | 检测 → 分类 → 处理 |
| 5 | 完整 Pipeline | overshoot (instrumented) | 完整日志 + 决策分析报告 |
| 6 | 测试套件 | - | pytest 运行所有恢复路径测试 |

## 文件结构

```
demo/
├── README.md              # 本文件
├── run_demo.sh            # 主 demo 脚本（有打字效果）
├── record_demo.sh         # asciinema 录制包装器
└── recordings/
    ├── exp-agent-demo.cast  # asciinema 录制文件
    └── exp-agent-demo.gif   # GIF 动画
```
