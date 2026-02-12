# Exp-Agent: 实验室硬件自主恢复代理

## 项目概述

**Exp-Agent** 是一个面向实验室硬件自主管理的恢复感知执行代理。它采用持续的"感知-决策-执行"循环，能够自动检测硬件故障、智能分类问题、并执行适当的恢复策略，无需人工干预。

### 核心理念

当实验室设备发生故障时，Exp-Agent 就像一位经验丰富的实验室技术员——感知问题、做出决策、安全地执行恢复操作，实现 24/7 全天候自主运行。

---

## 主要功能

### 1. 三阶段自主循环

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│  SENSE  │ ──→ │ DECIDE  │ ──→ │   ACT   │
│  感知   │     │  决策   │     │  执行   │
└────┬────┘     └─────────┘     └────┬────┘
     │                               │
     └───────────────────────────────┘
```

- **感知 (Sense)**: 监控设备状态，检测异常（温度过冲、超时、传感器故障、安全违规）
- **决策 (Decide)**: 智能分类错误，选择恢复策略
- **执行 (Act)**: 安全执行恢复动作，内置多层防护

### 2. 智能错误分类

| 错误类型 | 描述 | 恢复策略 |
|---------|------|---------|
| **可恢复 (Recoverable)** | 临时故障、瞬态错误（超时、通信问题） | 重试或降级 |
| **不可恢复 (Non-recoverable)** | 硬件损坏（传感器故障） | 安全关机 |
| **不安全 (Unsafe)** | 安全违规（温度过高、临界值） | 降级运行或中止 |

### 3. 四种决策类型

| 决策 | 场景 | 动作 |
|-----|------|-----|
| **RETRY** | 简单瞬态故障 | 重新尝试操作 |
| **SKIP** | 非关键步骤失败 | 跳过当前步骤，继续工作流 |
| **DEGRADE** | 性能受限 | 降低目标/要求，继续运行 |
| **ABORT** | 严重故障 | 执行安全关机 |

### 4. 多层安全执行

GuardedExecutor 执行器实施三层检查：

```
执行前 ─→ 安全检查 ─→ 执行后验证
  │           │            │
  ▼           ▼            ▼
验证前置条件  检查安全约束   轮询验证结果
             (如: 温度<130°C)
```

### 5. 工作流感知恢复

WorkflowManager 协调多步骤实验：
- **步骤依赖**: Critical / Hard / Soft / None
- **阶段管理**: Setup → SamplePrep → Measurement → Analysis → Cleanup
- **循环控制**: 支持多次实验迭代
- **上下文感知**: 基于当前状态做出恢复决策

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Exp-Agent 架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CLI 层 (run_sim.py, run_agent.py, run_workflow.py)        │
│       ↓                                                     │
│  编排层 (Orchestrator)                                      │
│  ├─ Supervisor (主控制循环)                                 │
│  └─ WorkflowManager (多步骤实验管理)                        │
│       ↓                                                     │
│  ┌──────────────────┬──────────────────┐                   │
│  ▼                  ▼                  ▼                   │
│ 执行器            恢复代理           设备层 (HAL)           │
│ ├─ GuardedExecutor  ├─ Classifier     ├─ SimHeater        │
│ └─ PostCheck        ├─ RecoveryAgent  ├─ RealDevice       │
│                     └─ WorkflowPolicy └─ Factory          │
│                                                             │
│  核心层 (Core)                                              │
│  ├─ Types (设备、动作、决策、错误)                          │
│  └─ Predicates (状态验证与监控)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心模块详解

### 1. Core 模块 (`core/`)

**types.py** - 基础数据结构：
- `DeviceState`: 设备遥测/状态数据
- `HardwareError`: 结构化错误（类型、严重程度、上下文）
- `Action`: 带有前置/后置条件和安全约束的操作
- `Decision`: 恢复决策（包含理由和动作列表）
- `ExecutionState`: 系统状态跟踪

**predicates.py** - 状态验证 DSL：
- 支持解析表达式如 `"temp ~= 120 +/- 2.0 within 20s"`
- 比较操作: `==`, `>`, `<`, `~=`, `one_of`
- 支持物理系统的容差比较

### 2. Devices 模块 (`devices/`)

**抽象设备接口**:
```python
class Device(ABC):
    def read_state(self) -> DeviceState: ...
    def execute(self, action: Action) -> bool: ...
    def health(self) -> HealthStatus: ...
```

**模拟加热器 (SimHeater)**:
- 基于物理的加热模拟
- 故障注入模式: `none`, `timeout`, `overshoot`, `sensor_fail`, `random`
- 动态温度更新与热力学模拟

**真实设备层**:
- 串口/网络连接管理
- 协议处理

### 3. Executor 模块 (`executor/`)

**GuardedExecutor** - 安全动作执行：
- 前置条件检查
- 安全约束验证
- 委托给 PostCheck 进行后置验证

**PostCheck** - 轮询验证：
- 阻塞直到后置条件满足
- 超时感知轮询
- 安全不变量强制执行（如温度始终 <130°C）
- 调试追踪收集

### 4. Recovery 模块 (`recovery/`)

**ErrorClassifier** - 错误分类：
- 将 HardwareError 映射到 ErrorProfile
- 返回可恢复性评估 + 推荐动作

**RecoveryAgent** - 决策制定：
- 分析错误档案 + 状态历史
- 检测模式（漂移、振荡、稳定）
- 返回带有理由的决策

**WorkflowPolicy** - 工作流感知恢复：
- WorkflowStep: 带依赖、阶段、重试次数的步骤
- ExperimentLoop: 跟踪循环状态和已完成步骤
- 上下文感知的恢复决策

### 5. Orchestrator 模块 (`orchestrator/`)

**Supervisor** - 主控制循环：
- 初始化设备 + 执行器 + 恢复代理
- 维护计划（动作序列）
- 处理重试预算和错误恢复
- 执行安全关机序列

**WorkflowManager** - 多循环实验：
- 按顺序执行工作流步骤
- 管理循环迭代（最多 max_loops 次）
- 跟踪步骤完成/失败
- 与恢复策略协调

---

## 数据流与执行循环

```
步骤 1: 观察 (OBSERVE)
  Device.tick() → Device.read_state() → DeviceState

步骤 2: 监控 (MONITOR)
  PostCheck.check_safety_invariants()
  → 检测异常 (temp > 130°C, sensor = -999, 等)
  → 如违反安全则抛出 HardwareError

步骤 3: 分类 (CLASSIFY)
  ErrorClassifier.classify(error) → ErrorProfile
  (可恢复? 不可恢复? 不安全?)

步骤 4: 决策 (DECIDE)
  RecoveryAgent.decide(state, error, history) → Decision
  (重试? 跳过? 降级? 中止?)

步骤 5: 执行恢复 (EXECUTE RECOVERY)
  GuardedExecutor.execute(recovery_action)
  ├─ 检查前置条件 ✓
  ├─ 检查安全约束 ✓
  ├─ Device.execute(action)
  └─ PostCheck.verify(postconditions) → 轮询循环

步骤 6: 验证并继续 (VALIDATE & CONTINUE)
  返回步骤 1 或进入下一个计划步骤
```

---

## 故障模式与场景

### 支持的故障模式

| 模式 | 描述 |
|-----|------|
| `none` | 正常路径（无故障） |
| `timeout` | 设备在 2 秒后无响应 |
| `overshoot` | 温度过度漂移 |
| `sensor_fail` | 传感器在 5 个 tick 后返回无效读数 |
| `random` | 随机注入故障 |

### 示例场景：温度过冲

```
[0s]  Agent: 设置目标温度 120°C
[5s]  Agent: 监控中... T=60°C ✓
[10s] Agent: 监控中... T=100°C ✓
[12s] Agent: ⚠️ 检测到过冲! T=128°C
      ├─ 分类: 可恢复 (temp < 安全限制 130°C)
      ├─ 决策: 降级 (降低目标至 110°C)
      └─ 执行: cool_down → set_temperature(110°C)
[15s] Agent: 已恢复，继续在 110°C 运行
[30s] Agent: 任务完成 ✓
```

---

## 配置说明

### 实验室配置 (`lab_config.json`)

```json
{
  "devices": [
    {
      "name": "heater_1",
      "type": "heater",
      "connection_type": "serial",
      "safety_limits": {"max_temperature": 250.0},
      "monitoring_interval": 1.0,
      "timeout": 5.0
    }
  ],
  "recovery_enabled": true,
  "max_retry_attempts": 3,
  "emergency_stop_on_critical": true
}
```

### 工作流配置 (`workflow_config.json`)

```json
{
  "steps": [
    {
      "name": "sample_heating",
      "phase": "sample_prep",
      "dependency": "soft",
      "can_skip": true,
      "max_retries": 3,
      "timeout": 600.0,
      "postconditions": ["sample_heated"]
    }
  ],
  "loop_control": {
    "max_loops": 5,
    "success_criteria": {"min_completed_steps": 7}
  }
}
```

---

## 使用指南

### 模拟模式（无需硬件）

```bash
# 正常运行
python -m exp_agent.cli.run_sim

# 测试过冲处理
python -m exp_agent.cli.run_sim --fault-mode overshoot

# 测试传感器故障处理
python -m exp_agent.cli.run_sim --fault-mode sensor_fail
```

### 真实硬件模式

```bash
# 串口连接
python -m exp_agent.cli.run_agent \
  --real-hardware \
  --device-name heater_1 \
  --port /dev/ttyUSB0 \
  --target-temp 120

# 网络连接
python -m exp_agent.cli.run_agent \
  --real-hardware \
  --device-name heater_1 \
  --host 192.168.1.100 \
  --target-temp 120

# 使用配置文件
python -m exp_agent.cli.run_agent \
  --real-hardware \
  --config my_lab_config.json
```

---

## 设计模式

| 模式 | 应用 |
|-----|------|
| **感知-决策-执行** | 主循环，清晰的关注点分离 |
| **基于谓词的状态验证** | 用于表达状态需求的 DSL |
| **基于策略的恢复** | 可插拔的恢复策略 |
| **多层安全** | 执行层的前置/后置/不变量检查 |
| **工厂模式** | 设备实例化抽象硬件类型 |
| **工作流编排** | 带依赖的步骤组合 |

---

## 依赖项

```
# 核心
pydantic>=2.0.0    # 类型验证

# 硬件
pyserial>=3.5      # 串口通信

# 可选
pyusb>=1.2.1       # USB 设备
pymodbus>=3.0.0    # Modbus 设备

# 测试
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

---

## 项目状态

### 当前功能 (MVP)

- ✅ 核心感知-决策-执行循环
- ✅ 错误分类（4 种类型）
- ✅ 4 种决策类型（重试/跳过/中止/降级）
- ✅ 三层安全检查的执行器
- ✅ 模拟加热器（5 种故障模式）
- ✅ 带循环的工作流管理
- ✅ 状态谓词 DSL

### 未来路线图

- 🔲 LLM 集成用于复杂分析
- 🔲 真实实验室设备集成
- 🔲 多设备协调
- 🔲 从故障历史中学习
- 🔲 高级诊断
- 🔲 分布式监控

---

## 目录结构

```
exp-agent/
├── exp_agent/
│   ├── __init__.py
│   ├── core/
│   │   ├── types.py          # 核心数据类型
│   │   └── predicates.py     # 状态验证 DSL
│   ├── devices/
│   │   ├── base.py           # 设备抽象接口
│   │   ├── factory.py        # 设备工厂
│   │   ├── simulated/
│   │   │   └── heater.py     # 模拟加热器
│   │   └── real/
│   │       └── device.py     # 真实设备
│   ├── executor/
│   │   ├── guarded_executor.py  # 安全执行器
│   │   └── post_check.py        # 后置验证
│   ├── recovery/
│   │   ├── classifier.py        # 错误分类器
│   │   ├── recovery_agent.py    # 恢复代理
│   │   └── workflow_policy.py   # 工作流策略
│   ├── orchestrator/
│   │   ├── supervisor.py        # 主控制器
│   │   └── workflow_manager.py  # 工作流管理器
│   └── cli/
│       ├── run_sim.py           # 模拟运行
│       ├── run_agent.py         # 代理运行
│       └── run_workflow.py      # 工作流运行
├── tests/                       # 测试文件
├── lab_config.example.json      # 配置示例
└── workflow_config.example.json # 工作流配置示例
```

---

## 总结

Exp-Agent 是一个用于实验室硬件管理的自主代理框架。它结合了强大的错误检测、智能分类和安全执行防护，实现实验室设备的无人值守运行。该架构模块化、可扩展，通过清晰的抽象层支持模拟和真实硬件。系统通过多层验证优先保障安全，同时通过基于策略的恢复决策保持灵活性。
