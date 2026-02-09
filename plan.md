# Lab Orchestrator Plan

更新日期：2026-02-08

## 1) 你的初始想法（原始目标）

你希望把 OpenClaw 的“常驻编排骨架”抽象成实验室可用 orchestrator，核心是三件事组合：

- Triggering（触发器/调度）
  - 时间触发：小时/天级 campaign loop
  - 事件触发：传感器越界、设备空闲、队列积压、QC fail
  - 外部触发：BO 推荐点、LIMS/ELN 新任务、人工审批通过
- Persistent State（持久状态）
  - source of truth 在 DB/对象存储
  - run、step、artifact、provenance、设备状态、告警全持久化
  - “summarization”落在可复用 run state/KPI/failure signature
- Session semantics / isolation（会话语义与隔离）
  - session = campaign_id / run_id / instrument_session
  - per-run sandbox（容器/进程/最小权限）避免单实验拖垮系统

你额外强调了实验室硬要求：

- 确定性与可重放（同 protocol + 输入 -> 同执行图）
- 安全闸门（interlock、SOP、阈值 gating）
- 并发与资源锁（仪器互斥、耗材与 slot 分配）
- 审计追责（审批、参数变更、firmware、calibration）

## 2) 我的执行计划（落地路线）

### 架构分层

- Control Plane：trigger ingress、session routing、workflow compile、scheduler、audit
- Execution Plane：per-run isolated worker、adapter 层、primitive runtime
- State Plane：DB + object store + 事件轨迹

### MVP 实施顺序

1. 建 FastAPI 项目骨架与核心模块边界
2. 建立持久化 schema（runs/steps/artifacts/provenance/locks/approvals/sessions）
3. 打通三类 trigger -> run 创建 -> 编译 -> preflight gate -> 调度状态
4. 实现 scheduler 常驻循环与 per-run subprocess worker
5. 接入模拟仪器 adapter（aspirate/heat/eis/wait/upload_artifact）
6. 加 runtime safety gate、资源锁、审计事件
7. 编写 README + 架构文档 + 基础测试与 smoke 验证

### 关键设计约束

- deterministic compile + graph_hash
- append-only 审计事件
- 资源锁采用 lease + fencing token
- 不以内存作为事实来源

## 3) 当前进度（状态跟踪）

### 已完成

- 项目骨架与依赖配置
  - `pyproject.toml`
  - `app/main.py`
- 持久化层（SQLite + schema）
  - `app/core/db.py`
- 编排核心
  - 协议编译与 graph_hash：`app/services/compiler.py`
  - preflight/runtime 安全闸门：`app/services/safety.py`
  - 资源锁管理：`app/services/lock_manager.py`
  - run/campaign 生命周期与审计：`app/services/run_service.py`
- 执行与隔离
  - per-run worker 子进程：`app/worker.py`
  - 调度器常驻循环：`app/services/scheduler.py`
- 触发与查询 API
  - campaigns：`app/api/v1/endpoints/campaigns.py`
  - triggers：`app/api/v1/endpoints/triggers.py`
  - runs/approval/events/locks：`app/api/v1/endpoints/runs.py`
- 模拟适配器与产物落盘
  - simulated instrument：`app/adapters/simulated_instrument.py`
  - object store：`app/services/artifact_store.py`
- 文档与测试
  - `README.md`
  - `docs/ARCHITECTURE.md`
  - `tests/test_compiler_and_safety.py`

- Bug 修复
  - compiler 增加循环依赖检测（Kahn 拓扑排序）：`app/services/compiler.py`
- 测试套件（52 tests, all passing）
  - `tests/test_compiler.py` — 16 tests（cycle detection, validation errors, determinism, edge cases）
  - `tests/test_safety.py` — 15 tests（boundary conditions, runtime interlocks, approval flag）
  - `tests/test_lock_manager.py` — 8 tests（lease/fencing token, contention, expiration, release idempotency）
  - `tests/test_api_integration.py` — 11 tests（full HTTP lifecycle: health, campaigns, triggers, rejection, approval, events）
  - `tests/test_compiler_and_safety.py` — 2 tests（original smoke tests）

- 真实硬件集成 — Stage 1: 适配器层 + 调度器改造（2026-02-08）
  - 从 refactored_battery 复制硬件模块到 `app/hardware/`
    - `dispatcher.py` — 26 个 action handler (ActionDispatcher)
    - `opentrons_controller.py` — OT-2 液体处理机器人
    - `plc_controller.py` — Modbus TCP 泵/搅拌器
    - `relay_controller.py` — SainSmart USB 16 通道继电器
    - `run_context.py` — 实验状态跟踪
    - `phase_result.py` — Phase 结果封装
  - 适配器协议 `app/adapters/base.py` — InstrumentAdapter Protocol
  - `app/adapters/simulated_instrument.py` — 重构为 SimulatedAdapter 类 + 向后兼容 execute_primitive()
  - `app/adapters/battery_lab.py` — BatteryLabAdapter (dry-run 模式 + 真实硬件模式)
  - `app/core/config.py` — 新增 adapter_mode / robot_ip / relay_port 等配置
  - `app/services/safety.py` — BATTERY_LAB_PRIMITIVES 扩展到 32 个 primitives
  - `app/services/run_service.py` — default_policy() 使用完整 primitives 列表
  - `app/worker.py` — 适配器工厂模式替代直接导入，adapter lifecycle (connect/disconnect)
  - `app/services/scheduler.py` — subprocess 改为 asyncio.to_thread 进程内线程执行
  - `app/services/compiler.py` — 支持 step_key 字段（兼容 translator 输出）
  - 修复所有硬件模块 import 路径（移除 config.constants / utils.logging_setup 依赖）

- 真实硬件集成 — Stage 2: 工作流翻译器（2026-02-08）
  - `app/services/workflow_translator.py` — phase-based JSON → OTbot flat DAG
    - 顺序 phase：步骤链式依赖
    - 并行 phase：thread 间无依赖，thread 内链式
    - 跨 phase：fork/join barrier
    - 资源自动映射（robot.* → ot2-robot 等）
  - `app/api/v1/endpoints/workflows.py` — POST /api/v1/workflows/import
  - `app/api/v1/router.py` — 注册 workflows endpoint

- 测试套件（93 tests, all passing — 2026-02-08）
  - `tests/test_adapters.py` — 16 tests（SimulatedAdapter + BatteryLabAdapter dry-run）
  - `tests/test_workflow_translator.py` — 25 tests（sequential/parallel/cross-phase/compiler compat/errors/resources）
  - （原有 52 tests 全部保持通过）

- 真实硬件集成 — Stage 3: 并行执行支持（2026-02-08）
  - `app/worker.py` — 并行执行循环
    - `_find_ready_steps()` 检测所有依赖已满足的就绪步骤
    - `_partition_by_resources()` 按资源分组，资源不重叠的步骤可并行
    - `_execute_step()` 线程安全的单步执行函数
    - 单步骤直接执行（无线程开销），多步骤 `threading.Thread` 并行
    - fork/join 语义：DAG 中的并行分支自动并行执行

- 真实硬件集成 — Stage 4: 错误策略 CRITICAL/BYPASS（2026-02-08）
  - `app/services/error_policy.py` — 错误分级模块
    - `CRITICAL_PRIMITIVES` — 14 个必须成功的操作（aspirate/dispense/labware loading/squidstat）
    - `BYPASS_PRIMITIVES` — 18 个可跳过的操作（homing/lights/relay/plots）
    - `classify_step_error(primitive, exc)` → "CRITICAL" | "BYPASS"
    - `ErrorPolicy` dataclass — 从 policy_snapshot 读取 `error_policy.allow_bypass`
  - `app/worker.py` — BYPASS 步骤标记为 "skipped" 而非 "failed"，不终止 run

- 真实硬件集成 — Stage 5: 产物集成（2026-02-08）
  - `app/services/artifact_store.py` — 新增 `persist_file_artifact()`
    - 支持 CSV/PNG 等二进制文件的 SHA-256 校验和计算
    - 复制到 object store 并保留原始扩展名

- 测试套件（155 tests, all passing — 2026-02-08）
  - `tests/test_error_policy.py` — 27 tests（CRITICAL/BYPASS 分类 + ErrorPolicy）
  - `tests/test_parallel_worker.py` — 17 tests（并行执行 + 文件产物）
  - （原有 93 tests + 新增 62 tests 全部通过）

- Phase B: Agent 工作区 + 能力注册表（2026-02-08）
  - `agent/SOUL.md` — Agent 身份与安全哲学（仿 OpenClaw SOUL.md 模式）
  - `agent/IDENTITY.md` — 名称、角色、实验室类型、仪器清单
  - `agent/AGENTS.md` — 操作手册：启动序列、Protocol 生命周期、错误策略、决策框架、子 Agent 规则
  - `agent/TOOLS.md` — 实验室环境配置（IP/端口/通道映射，与 skill 分离保护基础设施信息）
  - `agent/skills/robot.md` — OT-2 机器人 11 个 primitives（YAML frontmatter + 使用指南）
  - `agent/skills/plc.md` — PLC 泵/搅拌器 3 个 primitives
  - `agent/skills/relay.md` — 继电器 4 个 primitives（全 BYPASS）
  - `agent/skills/squidstat.md` — 电化学工作站 4 个 primitives
  - `agent/skills/utility.md` — 通用能力 8 个 primitives（wait/log/cleanup/sample/ssh）
  - `app/services/primitives_registry.py` — SKILL.md 解析器 + 内存索引 + LLM 摘要生成
  - `app/api/v1/endpoints/capabilities.py` — 4 个 API 端点
    - `GET /capabilities` — 完整技能目录
    - `GET /capabilities/primitives` — 按 instrument/error_class 过滤
    - `GET /capabilities/primitives/{name}` — 单个 primitive 详情
    - `GET /capabilities/summary` — LLM 友好的文本摘要
  - `tests/test_primitives_registry.py` — 28 tests（解析、查询、序列化、真实文件集成）
  - `tests/test_capabilities_api.py` — 7 tests（API 端点集成测试）

- 测试套件（190 tests, all passing — 2026-02-08）

### 已验证

- 语法编译通过：`python3 -m compileall app tests`
- 调度器自动领取并执行 run 的 smoke 测试通过（状态到 `succeeded`）
- `pytest` 全量通过：190 tests passed（2026-02-08）
- adapter_mode=simulated 模式下功能不变
- BatteryLabAdapter dry-run 模式下所有 26 个 action 通过
- 工作流翻译 → 编译 → graph_hash 一致性验证通过
- 并行执行：资源分组正确，fork/join DAG 语义正确
- 错误策略：CRITICAL/BYPASS 分类与 dispatcher.py 中的实际行为一致
- 文件产物：CSV/PNG 正确复制到 object store 并计算校验和

### 待完成

- 接真实硬件：设置 `ADAPTER_MODE=battery_lab`, `ADAPTER_DRY_RUN=false`
  - 确保 OT-2 IP 可达（`ROBOT_IP`）
  - 确保 PLC Modbus TCP 连接（默认 `192.168.1.2:502`）
  - 确保继电器串口可用（`RELAY_PORT`）
  - 确保 Squidstat SDK + Qt 事件循环（需专用 Qt 线程）
  - 安装硬件依赖：`opentrons`, `OT_PLC_Client_Edit`, `pyserial`, `SquidstatPyLibrary`
- 如需生产化：SQLite 升级 Postgres + 对象存储（S3/MinIO）

---

## 4) Agent Bot 路线图（从编排器到 AI 驱动的实验室 Agent）

### 启发来源：OpenClaw

OTbot 以 [OpenClaw](https://github.com/openclaw/openclaw) 为架构启发。OpenClaw 是一个本地优先的个人 AI 助手，核心模式与 OTbot 高度一致：

| 概念 | OpenClaw | OTbot 现状 | 对齐度 |
|------|----------|-----------|--------|
| 中央调度 | WebSocket Gateway | FastAPI + scheduler 循环 | ✅ |
| 触发路由 | 层级绑定分发 | trigger type 分发 (time/event/external) | ✅ |
| 并发控制 | Lane-based 隔离 | resource lock + `_partition_by_resources()` | ✅ |
| 插件/适配器 | Plugin 动态发现 | InstrumentAdapter Protocol + 工厂 | ✅ |
| 排队执行 | Chat Run Registry | `claim_schedulable_runs()` 原子领取 | ✅ |
| 人工审批 | Exec Approval Gate | `approvals` 表 + `approve_run()` API | ✅ |
| 会话隔离 | Session Key | session_key + instrument_sessions | ✅ |
| 持久状态 | SQLite | SQLite 8 张表 | ✅ |
| 审计追踪 | Session 管理 | append-only provenance_events | ✅ |
| 定时执行 | Cron Lane | campaigns + cadence_seconds | ✅ |
| LLM 推理 | 多 Provider 路由 + failover | — | ❌ 缺失 |
| 语义记忆 | SQLite-vec + FTS5 | — | ❌ 缺失 |
| 能力发现 | SKILL.md (LLM 可读的能力描述) | 硬编码 primitives 列表 | ❌ 缺失 |
| 自适应执行 | Agent 实时决策 | 静态 DAG | ❌ 缺失 |
| 实时通信 | WebSocket 双向 | HTTP 轮询 | ❌ 缺失 |

**结论**：OTbot 的编排骨架（"手和脚"）已完成。缺的是"大脑"（LLM）、"眼睛"（实时事件流）、和"记忆"（语义搜索）。

### 从 OpenClaw 学到的关键设计模式

#### Markdown-as-Agent-Rules 模式

OpenClaw 用 markdown 文件同时服务三个角色：
1. **人类文档**：开发者可直接阅读理解
2. **LLM 指令**：以第二人称写作，agent 直接读取执行
3. **机器配置**：YAML frontmatter 提供结构化元数据

**文件分职体系**（每个关注点独立一个文件）：

| 文件 | 用途 | OTbot 对应 |
|------|------|-----------|
| `SOUL.md` | Agent 人格/价值观/沟通风格 | `agent/SOUL.md` |
| `IDENTITY.md` | Agent 名称/角色/形象 | `agent/IDENTITY.md` |
| `AGENTS.md` | 行为规则手册（每次会话读取） | `agent/AGENTS.md` |
| `TOOLS.md` | 环境特定配置（本地设备信息） | `agent/TOOLS.md` |
| `USER.md` | 用户画像（偏好/项目/上下文） | `agent/USER.md` |
| `MEMORY.md` | 长期记忆（agent 自行维护） | `agent/MEMORY.md` |
| `BOOT.md` | 每次启动执行的任务 | `agent/BOOT.md` |
| `SKILL.md` | 每个能力的独立描述（含 frontmatter） | `agent/skills/*.md` |

**关键设计原则**：
- **子 Agent 过滤**：子 agent 只加载 AGENTS.md + TOOLS.md，不暴露个人数据
- **自修改工作区**：Agent 可编辑自己的 MEMORY.md / USER.md，形成进化型配置
- **生命周期分层**：BOOTSTRAP（首次） → BOOT（每次启动） → HEARTBEAT（定期轮询） → AGENTS（每次会话）
- **YAML frontmatter**：同一文件里结构化数据（给代码）和自然语言指令（给 LLM）共存不冲突
- **技能渐进披露**：frontmatter 始终可用，完整内容按需加载（控制 token 消耗）

### Agent 开发路线图

```
Phase A: 接真实硬件（已就绪，只需环境变量）     ✅
    ↓
Phase B: Agent 工作区 + 能力注册表               ✅ 完成（30 primitives, 5 skills, 4 API endpoints）
    ↓  （让 agent 能"看到"和"理解"仪器）
Phase C: SSE 实时事件流                         ← 下一步
    ↓  （让 agent 能实时观察实验进展）
Phase D: LLM Gateway（推理引擎）
    ↓  （agent 能从自然语言生成 protocol）
Phase E: 语义记忆
    ↓  （agent 能从过去实验中学习）
Phase F: 自适应执行 + 多 Agent
         （agent 能实时调整实验策略）
```

#### Phase B: Agent 工作区 + 能力注册表 ✅

**目标**：让 LLM 能够理解 OTbot 的所有仪器能力

**新建文件**：
- `agent/SOUL.md` — Agent 身份：实验室自动化 agent，谨慎、精确、安全第一
- `agent/IDENTITY.md` — 名称、角色、实验室类型
- `agent/AGENTS.md` — 行为规则手册：安全约束、决策框架、错误处理策略
- `agent/TOOLS.md` — 当前实验室硬件配置（IP、端口、labware 布局）
- `agent/skills/robot.md` — OT-2 机器人能力描述（11 个 primitives）
- `agent/skills/plc.md` — PLC 泵/搅拌器能力描述（3 个 primitives）
- `agent/skills/relay.md` — 继电器通道切换描述（4 个 primitives）
- `agent/skills/squidstat.md` — 电化学工作站描述（4 个 primitives）
- `agent/skills/utility.md` — wait/log/cleanup 等通用能力

**修改文件**：
- `app/services/safety.py` — primitives 列表从 skill markdown 的 frontmatter 生成
- 新增 `app/services/primitives_registry.py` — 解析 skill.md，提供 API 查询

**API**：
- `GET /api/v1/capabilities` — 返回所有注册的 primitives 及其 schema

#### Phase C: SSE 实时事件流

**目标**：实时推送实验状态给 UI 或 agent

**新建文件**：
- `app/api/v1/endpoints/events_stream.py` — SSE endpoint
- `app/services/event_bus.py` — 进程内事件总线（基于 asyncio.Queue）

**修改文件**：
- `app/services/audit.py` — `record_event()` 同时发布到 event_bus
- `app/worker.py` — 步骤开始/完成/失败事件发布

#### Phase D: LLM Gateway

**目标**：自然语言 → protocol 生成、实验结果解读

**新建文件**：
- `app/services/llm_gateway.py` — 多 Provider 抽象（Anthropic/OpenAI/local）+ failover
- `app/services/protocol_planner.py` — LLM 读取 skill.md → 生成 protocol dict
- `app/api/v1/endpoints/agent.py` — `POST /api/v1/agent/plan`

**关键设计**：
- 系统提示从 agent/*.md 文件组装（仿 OpenClaw bootstrap pipeline）
- skill.md 的 markdown body 注入 LLM context 作为工具描述
- Protocol 生成后走现有 compile → safety check → schedule 流程

#### Phase E: 语义记忆

**目标**：agent 能查询过去实验的结果和模式

**新建文件**：
- `app/services/memory.py` — 实验记忆管理（嵌入 + 索引）
- `agent/MEMORY.md` — agent 长期记忆（自行维护）

**修改文件**：
- `app/services/artifact_store.py` — 写入 artifact 时同步生成嵌入
- `app/core/db.py` — 新增 embeddings 表

#### Phase F: 自适应执行 + 多 Agent

**目标**：agent 能根据中间结果动态调整实验策略

**新增**：
- `agent.decide` primitive — 暂停执行，调用 LLM 决策
- 编译器支持条件边 (`depends_on_condition`)
- 多 Agent 角色（planner / executor / analyzer）各自有独立的 workspace md 文件

### 关键约束

- 每个 Phase 独立可验证，不依赖后续 Phase
- 现有 190 个测试在每个 Phase 完成后必须全部通过
- Phase B 是纯加法（新增文件 + 新增 API），不修改现有执行路径 ✅ 已验证
- LLM 永远不能绕过 safety gate —— compile → preflight → runtime 三层检查不可跳过
- Agent 生成的 protocol 与人类编写的 protocol 走完全相同的管道
