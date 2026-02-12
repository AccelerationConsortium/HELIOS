# Moltbot 项目分析：可借鉴的设计思想

## 概述

[Moltbot](https://github.com/moltbot/moltbot) 是一个本地优先的个人 AI 助手，支持多平台消息集成。虽然它的应用场景（个人助手）与 exp-agent（实验室故障恢复）不同，但其架构设计有很多值得借鉴的地方。

---

## 一、可借鉴的核心设计理念

### 1.1 Gateway 控制平面架构

**Moltbot 设计**：
```
┌─────────────────────────────────────────┐
│           Gateway Control Plane          │
│         (WebSocket: 127.0.0.1:18789)    │
├─────────────────────────────────────────┤
│  Sessions │ Channels │ Tools │ Events   │
└─────────────────────────────────────────┘
         ↑↓              ↑↓
    ┌────────┐      ┌────────┐
    │ Agents │      │ Nodes  │
    └────────┘      └────────┘
```

**借鉴到 exp-agent**：
```
┌─────────────────────────────────────────┐
│       Recovery Agent Control Plane       │
│         (WebSocket: localhost:8000)      │
├─────────────────────────────────────────┤
│ Experiments │ Devices │ Policies │ Logs  │
└─────────────────────────────────────────┘
         ↑↓              ↑↓
    ┌────────┐      ┌────────────┐
    │ Policy │      │ Hardware   │
    │ Engine │      │ Controllers│
    └────────┘      └────────────┘
```

**实施建议**：
- 将当前的 FastAPI WebSocket 升级为完整的控制平面
- 添加会话管理（Experiment Sessions）
- 支持多实验并行监控

---

### 1.2 分层的工具系统

**Moltbot 设计**：
- 工具是"一等公民"（First-class Tools）
- 工具分类：Browser、Canvas、Node、Automation
- 工具有独立的权限控制和沙箱策略

**借鉴到 exp-agent**：

```python
# 当前设计
@dataclass
class Action:
    name: str
    effect: Effect  # read | write
    params: dict

# 改进设计
@dataclass
class DeviceTool:
    """一等公民的设备工具"""
    name: str
    device_type: str  # heater | pump | positioner | spectrometer
    category: ToolCategory  # sensing | actuation | diagnostic | recovery
    effect: Effect
    params: dict

    # 新增：权限和安全控制
    requires_confirmation: bool = False  # 危险操作需要确认
    sandbox_policy: SandboxPolicy = SandboxPolicy.DEFAULT
    timeout_seconds: float = 30.0
    retry_policy: RetryPolicy = None
```

**工具分类体系**：
```yaml
tools:
  sensing:
    - read_temperature
    - read_flow_rate
    - read_position
    - read_spectrum

  actuation:
    - set_temperature
    - set_flow_rate
    - move_to_position
    - set_integration_time

  diagnostic:
    - run_self_test
    - calibrate
    - check_connection

  recovery:
    - cool_down
    - stop_pump
    - emergency_stop
    - home_position
```

---

### 1.3 配置驱动的 Agent 行为

**Moltbot 设计**：
```json
{
  "agent": {
    "model": "anthropic/claude-opus-4-5"
  },
  "dmPolicy": "pairing",
  "sandbox": {
    "mode": "docker",
    "allowlist": ["browser", "canvas"]
  }
}
```

**借鉴到 exp-agent**：

```yaml
# config/recovery_agent.yaml
agent:
  name: "Recovery Agent v1.0"
  mode: "autonomous"  # autonomous | supervised | manual

reasoning:
  model: "local-llm"  # 或 "gpt-4" | "claude" 用于复杂决策
  fallback: "rule-based"  # LLM 不可用时的回退策略

policies:
  error_profiles:
    overshoot:
      unsafe: true
      recoverable: true
      default_strategy: degrade
      max_retries: 3

    collision:
      unsafe: true
      recoverable: false
      default_strategy: abort
      requires_human_review: true

  recovery_strategies:
    degrade:
      backoff_schedule: [0, 2, 5, 10]
      degradation_factor: 0.9

    retry:
      max_attempts: 3
      exponential_backoff: true

safety:
  emergency_stop_enabled: true
  human_in_loop_threshold: "high"  # low | medium | high
  auto_abort_conditions:
    - "temperature > 200"
    - "pressure > 100"
    - "collision_detected"
```

---

### 1.4 会话状态管理

**Moltbot 设计**：
- 每个会话有独立状态（thinking level, verbose mode, model selection）
- 支持会话 patch 更新
- 会话隔离和组管理

**借鉴到 exp-agent**：

```python
@dataclass
class ExperimentSession:
    """实验会话状态管理"""
    session_id: str
    experiment_name: str

    # 实验状态
    status: Literal["setup", "running", "paused", "recovering", "completed", "failed"]
    current_step: int
    total_steps: int

    # 设备状态快照
    device_states: Dict[str, DeviceState]

    # 恢复上下文
    recovery_context: Optional[RecoveryContext] = None
    error_history: List[HardwareError] = field(default_factory=list)
    decision_log: List[Decision] = field(default_factory=list)

    # 配置
    thinking_level: Literal["fast", "careful", "deep"] = "careful"
    auto_recovery: bool = True
    human_approval_required: bool = False

    # 元数据
    started_at: datetime = None
    last_updated: datetime = None

    def patch(self, updates: Dict[str, Any]) -> None:
        """Moltbot 风格的状态 patch 更新"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.now()
```

---

### 1.5 多级故障恢复策略

**Moltbot 设计**：
- Model failover with OAuth/API key rotation
- Session pruning
- Retry policies with exponential backoff
- `moltbot doctor` 诊断命令

**借鉴到 exp-agent**：

```python
@dataclass
class RecoveryPolicy:
    """多级恢复策略"""

    # 第一级：自动重试
    retry_config: RetryConfig = field(default_factory=lambda: RetryConfig(
        max_attempts=3,
        backoff_schedule=[0, 2, 5, 10],
        exponential_backoff=True
    ))

    # 第二级：降级运行
    degrade_config: DegradeConfig = field(default_factory=lambda: DegradeConfig(
        degradation_factor=0.9,  # 目标值降低 10%
        max_degradations=2,      # 最多降级 2 次
        notify_operator=True
    ))

    # 第三级：安全停止
    abort_config: AbortConfig = field(default_factory=lambda: AbortConfig(
        safe_shutdown_required=True,
        preserve_sample=True,
        notify_operator=True,
        log_diagnostics=True
    ))

    # 诊断命令（类似 moltbot doctor）
    diagnostics: List[str] = field(default_factory=lambda: [
        "check_device_connections",
        "verify_calibration",
        "run_self_tests",
        "analyze_error_patterns"
    ])


def run_doctor() -> DiagnosticReport:
    """类似 moltbot doctor 的系统诊断"""
    report = DiagnosticReport()

    # 检查设备连接
    report.add_check("device_connections", check_all_devices())

    # 检查策略配置
    report.add_check("policy_config", validate_policies())

    # 检查安全阈值
    report.add_check("safety_thresholds", verify_safety_limits())

    # 检查日志系统
    report.add_check("logging_system", test_logging())

    # 检查恢复能力
    report.add_check("recovery_capability", simulate_recovery())

    return report
```

---

## 二、架构改进建议

### 2.1 当前架构 vs 改进架构

**当前架构**：
```
┌───────────┐     ┌──────────┐     ┌──────────┐
│ Dashboard │ ←→  │ FastAPI  │ ←→  │ Devices  │
│  (HTML)   │     │ WebSocket│     │ (Simulated)│
└───────────┘     └──────────┘     └──────────┘
                       ↓
                  ┌──────────┐
                  │ Policy   │
                  │ Engine   │
                  └──────────┘
```

**改进架构**（借鉴 Moltbot）：
```
                    ┌─────────────────────┐
                    │   Control Plane     │
                    │  (Gateway Service)  │
                    └─────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Session Mgr   │   │  Tool Registry│   │ Event Bus     │
│ - experiments │   │  - sensing    │   │ - errors      │
│ - states      │   │  - actuation  │   │ - telemetry   │
│ - history     │   │  - diagnostic │   │ - decisions   │
└───────────────┘   └───────────────┘   └───────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ↓
                    ┌─────────────────────┐
                    │   Policy Engine     │
                    │  (Rule + LLM Based) │
                    └─────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Heater      │   │    Pump       │   │ Spectrometer  │
│   Driver      │   │    Driver     │   │    Driver     │
└───────────────┘   └───────────────┘   └───────────────┘
```

---

### 2.2 具体实施路线图

#### Phase 1: 配置系统升级 (1-2 周)

```yaml
# 新增文件: config/agent_config.yaml
version: "1.0"

agent:
  name: "SDL Recovery Agent"
  mode: "autonomous"

devices:
  heater_1:
    type: heater
    connection: "serial:///dev/ttyUSB0"
    safety_limits:
      max_temperature: 200
      overshoot_threshold: 10

  pump_1:
    type: pump
    connection: "modbus://192.168.1.100"
    safety_limits:
      max_pressure: 100
      min_flow_rate: 0.1

policies:
  error_profiles:
    # 从 Python 迁移到 YAML
    overshoot:
      unsafe: true
      recoverable: true
      default_strategy: degrade

recovery:
  strategies:
    degrade:
      backoff_schedule: [0, 2, 5, 10]
    retry:
      max_attempts: 3
```

#### Phase 2: 会话管理系统 (2-3 周)

```python
# 新增文件: src/exp_agent/core/session.py

class SessionManager:
    """实验会话管理器"""

    def __init__(self):
        self.sessions: Dict[str, ExperimentSession] = {}
        self.event_bus = EventBus()

    async def create_session(self, experiment_config: dict) -> ExperimentSession:
        """创建新实验会话"""
        session = ExperimentSession(
            session_id=generate_id(),
            experiment_name=experiment_config["name"],
            status="setup"
        )
        self.sessions[session.session_id] = session
        await self.event_bus.emit("session_created", session)
        return session

    async def patch_session(self, session_id: str, updates: dict) -> None:
        """Moltbot 风格的会话更新"""
        session = self.sessions[session_id]
        session.patch(updates)
        await self.event_bus.emit("session_updated", session)
```

#### Phase 3: 工具注册系统 (2-3 周)

```python
# 新增文件: src/exp_agent/tools/registry.py

class ToolRegistry:
    """工具注册和管理系统"""

    def __init__(self):
        self.tools: Dict[str, DeviceTool] = {}
        self.categories: Dict[ToolCategory, List[str]] = defaultdict(list)

    def register(self, tool: DeviceTool) -> None:
        """注册设备工具"""
        self.tools[tool.name] = tool
        self.categories[tool.category].append(tool.name)

    def get_by_category(self, category: ToolCategory) -> List[DeviceTool]:
        """按类别获取工具"""
        return [self.tools[name] for name in self.categories[category]]

    async def execute(self, tool_name: str, params: dict,
                      session: ExperimentSession) -> ToolResult:
        """执行工具，带权限检查和超时控制"""
        tool = self.tools[tool_name]

        # 权限检查
        if tool.requires_confirmation and not session.human_approval_required:
            raise PermissionError(f"Tool {tool_name} requires human approval")

        # 执行带超时
        async with timeout(tool.timeout_seconds):
            result = await tool.execute(params)

        return result
```

#### Phase 4: LLM 集成 (3-4 周)

```python
# 新增文件: src/exp_agent/reasoning/llm_advisor.py

class LLMAdvisor:
    """LLM 辅助决策系统"""

    def __init__(self, config: LLMConfig):
        self.model = config.model
        self.fallback_to_rules = config.fallback

    async def analyze_error(self, error: HardwareError,
                           context: RecoveryContext) -> Analysis:
        """使用 LLM 分析错误"""
        prompt = self._build_prompt(error, context)

        try:
            response = await self.model.complete(prompt)
            return self._parse_analysis(response)
        except LLMError:
            # 回退到规则引擎
            return self.rule_engine.analyze(error, context)

    async def suggest_recovery(self, analysis: Analysis) -> List[RecoverySuggestion]:
        """LLM 建议恢复策略"""
        prompt = f"""
        Based on the error analysis:
        - Error Type: {analysis.error_type}
        - Signature: {analysis.signature}
        - Risk Level: {analysis.risk_level}

        Suggest recovery strategies ranked by likelihood of success.
        Consider: safety, sample preservation, experiment continuity.
        """

        response = await self.model.complete(prompt)
        return self._parse_suggestions(response)
```

---

## 三、具体功能借鉴清单

| Moltbot 功能 | exp-agent 借鉴 | 优先级 | 复杂度 |
|-------------|---------------|--------|--------|
| Gateway 控制平面 | WebSocket 控制中心 | 高 | 中 |
| 会话管理 | 实验会话状态 | 高 | 中 |
| 工具分类系统 | 设备工具注册 | 高 | 低 |
| YAML 配置 | 策略配置文件 | 高 | 低 |
| moltbot doctor | 系统诊断命令 | 中 | 低 |
| Retry policies | 多级恢复策略 | 高 | 中 |
| Event streaming | 事件流推送 | 中 | 中 |
| Sandbox policies | 安全执行沙箱 | 低 | 高 |
| Multi-channel | 多实验并行 | 低 | 高 |
| Model failover | LLM 回退机制 | 中 | 中 |

---

## 四、总结

从 Moltbot 可以借鉴的核心思想：

1. **控制平面架构**：将 Recovery Agent 设计为统一的控制中心
2. **一等公民工具**：设备操作作为可配置、可组合的工具
3. **配置驱动行为**：YAML 配置替代硬编码策略
4. **会话状态管理**：支持实验暂停、恢复、回放
5. **多级故障恢复**：retry → degrade → abort 的渐进式策略
6. **诊断命令**：类似 `moltbot doctor` 的系统健康检查

这些改进将使 exp-agent 从一个"Demo 展示系统"升级为"生产就绪的故障恢复平台"。
