# Exp-Agent Dashboard Demo 讲解稿

## 概述

这个 Dashboard 演示了一个用于自驱动实验室 (Self-Driving Lab, SDL) 的**故障恢复代理 (Recovery Agent)**。它展示了 Agent 如何在实验过程中检测硬件故障、分析故障模式、做出恢复决策，并执行相应的恢复动作。

---

## 一、系统架构

### 1.1 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                     Recovery Agent                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────┐  │
│  │ Devices │→ │  Policy  │→ │ Decision│→ │   Executor   │  │
│  │ (模拟)  │  │  Engine  │  │  Engine │  │              │  │
│  └─────────┘  └──────────┘  └─────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 文件结构

```
src/exp_agent/
├── core/types.py           # 核心数据类型定义
├── devices/simulated/      # 模拟设备
│   ├── heater.py          # 加热器 (温度控制)
│   ├── pump.py            # 泵 (流量控制)
│   ├── positioner.py      # 定位器 (XYZ运动)
│   └── spectrometer.py    # 光谱仪 (信号采集)
├── recovery/
│   └── policy.py          # 策略引擎 (核心决策逻辑)
└── web/app.py             # Web Dashboard
```

---

## 二、7 阶段恢复流水线

Dashboard 中间的流水线展示了 Agent 的决策过程，分为 7 个阶段：

### Stage 1: SENSE (感知)
**作用**: 从设备读取实时遥测数据

```python
# 读取设备状态
device_state = device.read_state()
# 返回: DeviceState(name="heater_1", status="error", telemetry={"temperature": 138.0})
```

**Dashboard 显示**: `→ temperature=138°C, status=error`

---

### Stage 2: CLASSIFY (分类)
**作用**: 将错误分类为可操作的配置文件 (ErrorProfile)

```python
@dataclass
class ErrorProfile:
    unsafe: bool              # 是否存在安全风险
    recoverable: bool         # 是否可恢复
    default_strategy: str     # 默认恢复策略
    safe_shutdown_required: bool
    diagnostics: List[str]    # 推荐诊断步骤
```

**分类逻辑 (当前为硬编码)**:

| 错误类型 | unsafe | recoverable | 默认策略 |
|---------|--------|-------------|---------|
| overshoot | True | True | DEGRADE |
| sensor_fail | True | False | ABORT |
| flow_blocked | True | True | DEGRADE |
| collision | True | False | ABORT |
| signal_saturated | False | True | DEGRADE |

**Dashboard 显示**: `→ error_type: overshoot` / `→ unsafe: True, recoverable: True`

---

### Stage 3: ANALYZE (分析)
**作用**: 分析遥测历史，识别故障模式 (Signature)

```python
SignatureMode = Literal["drift", "oscillation", "stall", "noisy", "stable", "unknown"]

def analyze_signature(history: List[DeviceState], metric: str) -> SignatureResult:
    # 计算特征
    # - drift: 连续上升/下降趋势
    # - oscillation: 周期性波动
    # - stall: 数值停滞不变
    # - noisy: 高方差随机波动
    # - stable: 正常稳定状态
```

**检测算法**:
- **Drift**: 计算相邻点斜率，若持续正/负则判定为漂移
- **Stall**: 若连续 N 个采样点变化 < ε，判定为停滞
- **Oscillation**: 检测峰谷交替，计算振幅

**Dashboard 显示**: `→ pattern: drift (continuous rise)` / `→ delta: +18°C over target`

---

### Stage 4: DECIDE (决策)
**作用**: 基于 ErrorProfile + SignatureResult 做出恢复决策

**决策类型**:

| 决策 | 含义 | 触发条件 |
|-----|------|---------|
| **ABORT** | 终止实验 | unsafe=True 且 recoverable=False |
| **DEGRADE** | 降级运行 | unsafe=True 但 recoverable=True |
| **RETRY** | 重试操作 | unsafe=False 且为暂时性错误 |
| **SKIP** | 跳过步骤 | 非关键步骤失败 |

**决策逻辑 (policy.py)**:
```python
def make_recovery_decision(error, history, target, retry_count):
    profile = classify_error(error)
    signature = analyze_signature(history)

    # 1. 不可恢复 → ABORT
    if not profile.recoverable:
        return RecoveryDecision(kind="abort", ...)

    # 2. 可恢复 + 未超重试 → RETRY
    if retry_count < max_retries:
        return RecoveryDecision(kind="retry", ...)

    # 3. 重试耗尽 → DEGRADE
    degraded_target = compute_degraded_target(target, signature.mode)
    return RecoveryDecision(kind="degrade", ...)
```

**Dashboard 显示**:
- `→ Option 1: ABORT - lose sample`
- `→ Option 2: DEGRADE - reduce to 110°C ✓`
- `DECISION: DEGRADE → set_temperature(110°C)`

---

### Stage 5: EXECUTE (执行)
**作用**: 执行恢复动作

**恢复动作示例**:
```python
def cool_down() -> Action:
    return Action(name="cool_down", effect="write")

def set_temperature(temp: float) -> Action:
    return Action(name="set_temperature", effect="write", params={"temperature": temp})

def wait_action(seconds: float) -> Action:
    return Action(name="wait", effect="write", params={"duration": seconds})
```

**Dashboard 显示**: `→ cool_down() initiated` / `→ target reached: 110°C`

---

### Stage 6: VERIFY (验证)
**作用**: 检查后置条件是否满足

```python
# 验证恢复是否成功
new_state = device.read_state()
assert new_state.status != "error"
assert new_state.telemetry["temperature"] <= 130  # 安全阈值
```

**Dashboard 显示**: `→ temperature stable at 110°C ✓` / `→ status=running ✓`

---

### Stage 7: MEMORY (记忆)
**作用**: 将经验记录到数据库，用于未来学习

```python
experience = {
    "error_type": "overshoot",
    "signature": "drift",
    "decision": "degrade",
    "outcome": "success",
    "context": {...}
}
# 存入经验数据库，供未来 LLM 学习
```

**Dashboard 显示**: `→ pattern: overshoot → degrade` / `→ outcome: SUCCESS`

---

## 三、4 种模拟设备

### 3.1 Heater (加热器)
- **遥测**: temperature, target_temp
- **故障模式**: overshoot, sensor_fail, safety_violation
- **恢复动作**: set_temperature(), cool_down()

### 3.2 Pump (泵)
- **遥测**: flow_rate, pressure
- **故障模式**: flow_blocked, pressure_drop, leak_detected, cavitation
- **恢复动作**: stop_pump(), prime_pump(), reduce_flow()

### 3.3 Positioner (定位器)
- **遥测**: x, y, z position
- **故障模式**: collision, position_drift, motor_stall, encoder_error
- **恢复动作**: stop(), home(), retract()

### 3.4 Spectrometer (光谱仪)
- **遥测**: signal_intensity, integration_time
- **故障模式**: signal_saturated, baseline_drift, lamp_failure
- **恢复动作**: reduce_integration(), dark_subtract(), recalibrate()

---

## 四、Policy 设计：硬编码 vs YAML

### 当前状态：硬编码 (Python)

目前 Policy 是**硬编码在 Python 中**的，位于 `recovery/policy.py`：

```python
# 错误分类 - 硬编码
def classify_error(error: HardwareError) -> ErrorProfile:
    if error.type == "overshoot":
        return ErrorProfile(unsafe=True, recoverable=True, default_strategy="degrade")
    if error.type == "collision":
        return ErrorProfile(unsafe=True, recoverable=False, default_strategy="abort")
    # ... 更多规则
```

**优点**:
- 类型安全，IDE 支持好
- 逻辑清晰，易于调试
- 测试覆盖方便

**缺点**:
- 修改需要改代码
- 非技术人员难以配置

### 未来计划：YAML 配置 (TODO)

代码中已预留 YAML 配置接口：

```python
# policy.py 第 58-59 行
# Global configs (will load from YAML later)
SIGNATURE_CONFIG = SignatureConfig()
RECOVERY_CONFIG = RecoveryConfig()
```

**计划的 YAML 格式**:

```yaml
# recovery_policy.yaml
error_profiles:
  overshoot:
    unsafe: true
    recoverable: true
    default_strategy: degrade
    safe_shutdown_required: true

  collision:
    unsafe: true
    recoverable: false
    default_strategy: abort

signature_config:
  drift_slope_threshold: 0.5
  stall_epsilon: 0.1
  oscillation_amplitude_threshold: 2.0

recovery_config:
  backoff_schedule: [0, 2, 5, 10]
  max_retries_per_error: 3
  default_degraded_temp: 110.0
```

**迁移计划**:
1. 保留 Python dataclass 作为类型定义
2. 添加 YAML 加载器
3. 支持运行时热加载
4. 保留 Python 覆盖能力（复杂逻辑）

---

## 五、Demo 场景详解

### Scenario 1: Heater Overshoot → DEGRADE

**故障**: 加热器超调，温度 138°C > 130°C 限制
**分析**: drift 模式（持续上升）
**决策**: DEGRADE - 降低目标温度至 110°C
**结果**: 实验继续，数据质量可能受影响

### Scenario 2: Pump Blockage → DEGRADE

**故障**: 流量从 50 mL/min 降至 0（堵塞）
**分析**: drift 模式（持续下降）+ 压力上升
**决策**: DEGRADE - 停泵 + 冲洗
**结果**: 需要人工检查管路

### Scenario 3: Positioner Collision → ABORT

**故障**: 定位器在 x=15mm 处碰撞
**分析**: stall 模式（运动停止）
**决策**: ABORT - 不可恢复的物理损坏
**结果**: 实验终止，样品标记为 COMPROMISED

### Scenario 4: Spectrometer Saturation → DEGRADE

**故障**: 信号 65000 超过 16-bit 最大值
**分析**: drift 模式（信号持续上升）
**决策**: DEGRADE - 减少积分时间 50%
**结果**: 信号恢复至有效范围，继续采集

---

## 六、关键设计原则

### 6.1 安全优先 (Safety First)
- `unsafe=True` 时必须执行 `safe_shutdown_required` 检查
- 物理碰撞等不可恢复故障直接 ABORT

### 6.2 渐进式恢复 (Progressive Recovery)
```
RETRY (轻量) → DEGRADE (中度) → ABORT (最终)
```

### 6.3 可观测性 (Observability)
- 每个阶段都有日志输出
- 决策理由 (rationale) 被记录
- 支持事后分析

### 6.4 样品状态追踪 (Sample Status)
```python
SampleStatus = Literal["intact", "compromised", "destroyed", "anomalous"]
```

---

## 七、下一步开发计划

1. **Phase 2**: 接入真实 LLM 进行决策辅助
2. **Phase 3**: 经验数据库 + 基于案例的推理
3. **Phase 4**: 多设备协调恢复
4. **Phase 5**: YAML Policy 配置化

---

## 附录：运行 Dashboard

```bash
# 安装依赖
uv add fastapi uvicorn websockets

# 启动 Dashboard
uv run python -m exp_agent.web.app

# 打开浏览器
open http://localhost:8000
```

**交互**:
- 点击设备按钮注入特定故障
- 点击 "Run All Scenarios" 运行完整演示
- 观察 Event Log 查看详细推理过程
