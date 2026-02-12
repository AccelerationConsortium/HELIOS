# Sensing Layer Demo

感知层演示 - 展示"盲目恢复"与"感知驱动恢复"的区别。

## Quick Start

### 一键生成所有视频（推荐）

```bash
# 安装依赖
brew install asciinema agg ffmpeg

# 一键生成所有 demo 的 .cast / .gif / .mp4
./demo/generate_sensing_videos.sh

# 输出文件:
# demo/recordings/sensing-demo-1.{cast,gif,mp4}
# demo/recordings/sensing-demo-2.{cast,gif,mp4}
# demo/recordings/sensing-demo-3.{cast,gif,mp4}
# demo/recordings/sensing-demo-4.{cast,gif,mp4}
# demo/recordings/sensing-demo-all.{cast,gif,mp4}
```

### 手动录制

```bash
# 录制单个 demo
./demo/record_sensing_demo.sh --demo 1

# 录制全部 demo
./demo/record_sensing_demo.sh

# 录制 + 自动转换
./demo/record_sensing_demo.sh --convert
```

### 直接运行（不录制）

```bash
# 运行所有演示（带打字效果）
./demo/run_sensing_demo.sh

# 运行特定演示
./demo/run_sensing_demo.sh --demo 1  # Blind vs Sensing-Aware
./demo/run_sensing_demo.sh --demo 2  # Real-time Sensor Panel
./demo/run_sensing_demo.sh --demo 3  # Incident Replay
./demo/run_sensing_demo.sh --demo 4  # SafetyAdvisor Integration

# 快速模式（跳过延迟）
./demo/run_sensing_demo.sh --fast
```

### 回放录制

```bash
# 正常速度
asciinema play demo/recordings/sensing-demo-1.cast

# 2倍速
asciinema play --speed 2 demo/recordings/sensing-demo-all.cast
```

## 演示内容

### Demo 1: Blind vs Sensing-Aware Recovery（主打）

**场景**: 同一个故障，用不用感知层，系统行为完全不同

- **Step A (旧系统)**: 设备超时 → 盲目 RETRY → 看似恢复成功
- **Step B (新系统)**: 设备超时 + 温度过高 + 通风不足 → **VETO** 阻止恢复

**💥 视觉冲击点**: 同一个错误，以前会继续跑，现在被"现实世界状态"拦住

### Demo 2: Real-time Sensor Panel

**场景**: 实时传感器面板 + 联锁触发

- 左边: 实时 sensor readings (温度/斜率/风速/压力)
- 右边: SafetyState 状态机 (SAFE → DEGRADED → INTERLOCKED)
- 看到状态从绿到黄到红的渐变

**💡 展示点**: 这叫"闭环控制"，不是 agent 写作文

### Demo 3: Incident Replay

**场景**: 故障回放 + 复盘分析

```
replay --log incident_2026_02_05.json
```

- 时间轴回放触发事件
- 根因分析输出
- 确定性证据链验证

**💡 展示点**: 工程可信度直接拉满

### Demo 4: SafetyAdvisor Integration

**场景**: 当进入 INTERLOCKED 状态时，SafetyAdvisor 提供领域专业解释

- 结合当前化学品（甲苯 + 氢化钠）
- 分析蒸汽累积风险
- 提供具体恢复建议

**💡 展示点**: 解释"为什么"被阻止，而不只是"被阻止了"

## 核心概念展示

| 概念 | 展示内容 |
|------|----------|
| SensorEvent | 标准化传感器数据格式 |
| SafetyStateMachine | SAFE → DEGRADED → INTERLOCKED → EMERGENCY |
| EvidenceChain | snapshot_id + trigger_events 证据链 |
| RecoveryGate | "无盲目恢复"规则 |
| SafetyAdvisor | 建议性响应，无可执行动作 |

## 关键消息

1. **以前**: Agent 只知道"出错了"→ 盲目重试
2. **现在**: Agent 看到"现实世界状态"→ 智能决策
3. **证据链**: 每个决策都可审计、可重放
4. **安全第一**: 高风险动作必须传感器验证
