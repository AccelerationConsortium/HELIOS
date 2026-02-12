把 alan 组这个“计算边界”的化学安全 agent，接到你“执行边界”的灾难恢复 agent，核心就是：让安全 agent 给你的执行策略加一个“化学风险上下文”，并把它变成可执行的 gate / policy / recovery 约束。这套安全 agent 本身就是“StartNode→路由→多个专业 Agent→汇总报告”的多智能体架构，输出里也明确包含 hazard / PPE / SOP / 阈值 / 应急 这些你最需要落地成规则的东西。 ￼


1) 角色分工：谁管什么边界
	•	Alan 安全 agent（计算边界）：回答“这个实验/化学品危险吗、需要什么 PPE、SOP 怎么写、温度/速率阈值是多少、泄漏/接触怎么应急”。它的输出形态就已经包含 GHS、PPE、SOP、监控阈值、应急表格。 ￼  ￼
	•	你的灾难恢复 agent（执行边界）：回答“设备超时/掉线/卡阀/泵堵/机械臂碰撞/传感器异常时，怎么自动重试、怎么回滚、怎么进入安全停机、怎么恢复到可继续的状态”。

结合目标：执行侧所有 recovery 动作都必须满足化学安全侧给出的约束（比如“不能升温”“必须通风橱”“必须撤离”“不得继续加料”等）。

⸻

2) 结合点一：Pre-flight Safety Gate（开跑前卡口）

在 workflow/campaign 启动前，加一个固定步骤：

(A) 生成“实验意图摘要”（机器可读，不是作文）
	•	化学品列表（CAS/SMILES/名称）
	•	步骤序列（加料、搅拌、加热、抽真空、放气等）
	•	关键可控参数范围（温度、压力、加料速率、浓度等）
	•	设备与环境（通风橱/封闭系统/是否有洗眼器等）

(B) 调用安全 agent 产出 SafetyPacket
安全 agent 支持自然语言和 API 调用，且明确支持 CAS/SMILES 等结构化输入。 ￼
它还能生成完整 SOP（包含阈值与应急）。 ￼

(C) Gate 决策
	•	ALLOW：写入 run context，允许执行
	•	ALLOW_WITH_CONSTRAINTS：写入“硬约束”，执行侧必须遵守
	•	DENY：阻止启动，要求人工确认/修改方案

你想要的效果：Willi 说“安全第一”时你能给他一个真的“第一”，不是口号。

⸻

3) 结合点二：Run-time Safety Overlay（执行时的安全覆盖层）

把 SafetyPacket 里的内容变成执行时实时可用的约束与动作映射。

(1) 从安全 agent 提取“可执行信号”

安全 agent 的 SOP 输出里有你最喜欢的东西：监控项 + 安全阈值 + 超阈动作（例子里甚至写了“温度升高 >3°C/min 立即停止”这种可直接变 policy 的句子）。 ￼

抽成这几类字段（建议你就用 JSON schema 固化）：
	•	hazards: GHS/H statements
	•	ppe: 必备 PPE
	•	monitoring: 需要监控的变量列表
	•	thresholds: 变量阈值与严重级别
	•	emergency_playbooks: 情况→措施（泄漏、接触、火灾等）

(2) 灾难恢复 agent 的动作要“过安全审计”

给你的 RecoveryAction 加一个检查器：
	•	action = RETRY / FLUSH / REHOME / DRAIN / VENT / QUENCH / SHUTDOWN / EVACUATE / ASK_HUMAN
	•	safety_check(action, SafetyPacket, current_state) -> ALLOW / BLOCK / REQUIRE_HUMAN

例子：
	•	设备超时：一般允许 AUTO-RETRY
	•	但如果 SafetyPacket 标注“易燃溶剂 + 密闭系统 + 需要通风橱”，那 “打开加热 / 提升搅拌 / 增加加料速率” 这类动作可能直接 BLOCK
	•	若检测到“泄漏/接触”类别事件，直接跳 EVACUATE（安全 agent 的应急场景就覆盖了这个决策逻辑）。 ￼

(3) 把“问答安全系统”变成你的“异常解释器”

安全 agent 设计里明确支持“实验中异常就问：温度超过50°C怎么办”。 ￼
你可以在灾难恢复 agent 里用它做两件事：
	1.	为异常事件生成 human-readable 的解释与建议
	2.	为策略选择提供额外证据（例如：是否需要立即停机、是否需要通风、是否存在不相容化学品）

⸻

4) 结合点三：共享事件总线与最小接口

你不需要把两个系统揉成一个 repo。用最小接口耦合就行：

SafetyAgent 提供
	•	assess(plan) -> SafetyPacket
	•	answer(question, context) -> guidance

它本来就是多 agent 路由结构。 ￼

RecoveryAgent 发布 / 订阅
	•	发布：IncidentEvent（timeout、spill suspected、overheat、pressure anomaly…）
	•	订阅：SafetyPacket（开跑前一次）+ guidance（需要时按事件触发）

⸻

5) 最关键的一步：统一“严重级别”与“谁有最终否决权”

建议你硬规定：
	•	化学安全事件（spill/exposure/fire/incompatible mix/overheat beyond threshold）：安全 agent 或其规则层拥有 最终否决权，RecoveryAgent 只能进入 SAFE_SHUTDOWN/EVACUATE/ASK_HUMAN。
	•	纯设备事件（瞬时 timeout/短暂断连/机械臂 rehome）：RecoveryAgent 主导，但动作要过 safety overlay。

这样边界清晰，不会出现“为了恢复设备把实验室点了”这种人类经典操作。

⸻

以上就是结合方案：Pre-flight 卡口 + Run-time 覆盖层 + 最小接口 + 否决权规则。你要是照这个做，alan 组那套就不再是“问答玩具”，而是你灾难恢复 agent 的“安全约束内核”。

0205
当前完成状态
✅ Phase 1 - 完成 (Safety Agent 基础集成)
组件	状态	说明
Safety 类型系统	✅	SafetyPacket 完整 schema (GHS, PPE, 阈值, 约束)
SafetyAgent Protocol	✅	最小接口: assess() + answer()
Mock SafetyAgent	✅	4种化学品配置, 可配置响应
Safety Checker	✅	Action 验证逻辑
GuardedExecutor 集成	✅	3层防护 + 化学安全检查
RecoveryAgent 集成	✅	化学安全事件的 veto 权
WorkflowSupervisor 集成	✅	Pre-flight gate + packet 传递
错误分类扩展	✅	12种化学安全错误类型
配置文件	✅	config/safety.yaml 完整
🔲 Phase 2 - Scaffolding (LLM Advisor)
LLMAdvisor protocol 已定义
"Proposal-only" 模式: LLM 只提建议，policy 仍然决策
目前只有 stub 实现
下一步开发计划
根据 plan.md 和 next_plan.md，建议的优先级：

🎯 近期 (Phase 2 完善)
LLM Advisor 真实实现

连接实际 LLM (Claude/GPT)
实现 propose_recovery() 带约束空间
添加 confidence 评估
SafetySDLAgent 适配器

连接 Alan 的真实 Safety Agent
替换 MockSafetyAgent
实现 API 客户端
测试覆盖扩展

test_safety_integration.py 需要更多场景
端到端集成测试
Veto 规则边界测试
📋 中期 (Phase 3)
Emergency Playbook 触发

Playbook 定义已有，但未集成到决策逻辑
实现 skin_contact, spill, fire 等响应流程
Anomaly Detection 完善

区分 scientific outliers vs errors
实现 quarantine + replicate 流程
多设备协调

当前假设单一 heater
扩展到多设备场景

0205 计划

把“感知层”做成一个可插拔、可模拟、可审计的数据面板，先别急着上 LLM。

感知层目标

把各种传感器/信号统一成一条标准事件流，给上层（SafetyChecker / RecoveryAgent / LLMAdvisor）消费：
	•	统一读数格式（时间戳、单位、质量标记、来源）
	•	健康度（断连、漂移、噪声、卡死、超量程）
	•	状态估计（可选，先留接口）
	•	可回放（事故复盘必须能重放）

⸻

1) 先定“要接的信号清单”


P0（最先做）
	•	温度（反应器/热板/箱体）
	•	压力（密闭系统）
	•	通风橱风速/状态（开关、风速是否达标）
	•	设备电源/急停状态（E-stop / mains）

P1
	•	气体/VOC/可燃/有毒（取决于实验）
	•	漏液/液位（地面漏液、托盘液位）
	•	门禁/区域占用（最简版就是“门磁/开关”）

P2
	•	摄像头视觉（液体溢出、烟雾、人员进入）
	•	人员位置（能不做就先不做，隐私和部署都麻烦）

⸻

1) 定一个“SensorEvent”标准（核心资产）

所有输入，不管来自串口、MQTT、Modbus、HTTP，最后都变成这个：

{
  "ts": "2026-02-05T14:20:11.123Z",
  "sensor_id": "hood_01_airflow",
  "type": "airflow_mps",
  "value": 0.42,
  "unit": "m/s",
  "quality": {
    "status": "OK | STALE | DROPPED | OUT_OF_RANGE | CALIBRATION_DUE",
    "confidence": 0.0-1.0
  },
  "meta": {
    "location": "SDL1_hood_A",
    "source": "modbus|mqtt|serial|http",
    "raw": "optional"
  }
}

上层只认这一个格式。别让上层知道“这玩意是串口读的还是 PLC 推的”，否则你未来会被自己气死。

⸻

3) 感知层组件拆分（可插拔）

A) Sensor Driver（适配器）

每种来源一个 driver，职责很纯：
	•	采集
	•	解析
	•	单位/量程归一
	•	输出 SensorEvent

B) Sensor Hub（聚合器）
	•	统一时间戳（本机 monotonic + wall clock）
	•	去重、限频、缓存（ring buffer）
	•	写入事件总线（Redis Streams / NATS / Kafka 选一个轻的先）

C) Health Monitor（必做）

对每个 sensor_id 维护：
	•	last_seen
	•	expected_period
	•	dropout_rate
	•	stuck_value_detect
	•	drift/variance（可选）

输出 SensorHealthEvent，给 SafetyChecker 用。

D) Snapshot API（给决策层拉取状态）

提供 GET /sensors/snapshot 返回“当前状态面板”：
	•	每个传感器最近值 + health
	•	最近 N 秒的简短窗口（用于趋势判断）

⸻

4) 先做“模拟器”而不是先买硬件

你已经有 simulation mode 的思路，这里直接延伸：
	•	MockSensorDriver：从脚本/CSV/函数生成读数
	•	FaultInjector：丢包、噪声、卡死、漂移、突变
	•	能跑回放：把真实 log 喂回去重现事故

这样你能立刻把感知层接到你的 GuardedExecutor/RecoveryAgent 上做端到端测试。

⸻

5) 第一版的“感知→安全联锁”落地路径

别一上来就搞视觉模型和风险评估。先把最硬的几条联锁跑通：
	•	hood airflow < 阈值 → veto：禁止启动/继续挥发性步骤；进入 safe shutdown
	•	temperature slope > 阈值 → veto：停止加热/停止加料；报警 + 人工介入
	•	pressure > 阈值 → veto：停止加热/停止加料；执行泄压流程（如果你有软联锁动作）
	•	sensor stale（超过 2× expected period）→ 降级：禁止进入高风险步骤

这些都是你现有 SafetyChecker 能直接消费的东西。

⸻

6) 你今天就能开始写的目录结构

sensing/
  protocol/
    sensor_event.ts
    health_event.ts
    snapshot.ts
  drivers/
    mock_driver.ts
    modbus_driver.ts
    mqtt_driver.ts
    serial_driver.ts
    http_driver.ts
  hub/
    sensor_hub.ts
    ring_buffer.ts
  health/
    health_monitor.ts
    detectors/
      stale.ts
      stuck.ts
      out_of_range.ts
  simulator/
    fault_injector.ts
    replay.ts
  api/
    sensing_api.ts   # /snapshot, /health


⸻

7) 你做对了会立刻得到什么
	•	RecoveryAgent 不再是“对着空气推理”
	•	你的 veto 不再只是化学知识，而是基于现场状态
	•	LLMAdvisor 以后接入也变得安全：它看的是 snapshot，而不是让它去“想象”传感器

先把感知层跑通，你的系统就从"会写 SOP 的机器人"变成"至少能看见现实的机器人"。这一步非常值。

---

## 0205 实施完成

### ✅ 感知层实现 (2026-02-05)

已完成 0205 计划中的感知层核心架构：

**目录结构**:
```
sensing/
├── protocol/
│   ├── sensor_event.py      # ✅ SensorEvent 核心类型
│   ├── health_event.py      # ✅ SensorHealthEvent
│   └── snapshot.py          # ✅ SensorSnapshot, SystemSnapshot
├── drivers/
│   ├── base.py              # ✅ SensorDriver 基类
│   └── mock_driver.py       # ✅ MockSensorDriver + 实验室传感器集
├── hub/
│   ├── sensor_hub.py        # ✅ 聚合器 (去重、限频、缓存)
│   └── ring_buffer.py       # ✅ 时间窗口环形缓冲
├── health/
│   ├── health_monitor.py    # ✅ 健康监控器
│   └── detectors/
│       ├── stale.py         # ✅ 超时检测
│       ├── stuck.py         # ✅ 卡值检测
│       └── out_of_range.py  # ✅ 越界检测
├── simulator/
│   ├── fault_injector.py    # ✅ 故障注入器 (dropout, stuck, drift, spike)
│   └── replay.py            # ✅ 事件回放
├── api/
│   └── sensing_api.py       # ✅ REST API (/snapshot, /health, /sensors/{id})
└── integration.py           # ✅ 与 SafetyChecker/RecoveryAgent 集成
```

**核心功能**:
1. **SensorEvent 标准**: 所有输入统一格式 (ts, sensor_id, type, value, unit, quality, meta)
2. **P0 传感器支持**: 温度、压力、通风橱风速、急停状态
3. **健康检测**: stale (超时)、stuck (卡值)、out_of_range (越界)
4. **故障模拟**: MockSensorDriver + FaultInjector 支持端到端测试
5. **安全联锁集成**: SensingIntegration 提供实时 veto 检查

**安全联锁实现** (plan.md section 5):
- `hood airflow < 0.3 m/s` → veto: 禁止启动/继续挥发性步骤
- `temperature > 130°C` → veto: 停止加热
- `temperature slope > 3°C/min` → veto: 停止加热，报警
- `pressure > 200 kPa` → veto: 停止加热，执行泄压
- `sensor stale (> 2x expected period)` → degrade: 禁止进入高风险步骤

**测试覆盖**: 21 tests passing

### 下一步计划

1. **真实驱动实现**: Modbus/MQTT/Serial 驱动
2. **与现有 SafetyChecker 深度集成**: 替换 telemetry 数据源
3. **与 RecoveryAgent 集成**: 传感器事件触发恢复决策
4. **仪表板 UI**: 实时传感器状态可视化