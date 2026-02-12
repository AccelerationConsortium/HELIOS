要结合 Alan 组那两个 agent，你现在这套 **Execution-layer disaster recovery** 不需要推翻；你需要做的是：把它放进一个 **3-layer agent stack**，并把“对上对下”的接口都做成稳定的 contract。这样 Alan 组的 agent（大概率是更偏 decision / scientist / tool-using 的）可以把你当成一个可靠的“执行底座”。

下面我给你一个具体结合方式：**位置、接口、数据流、最小实现**。我会假设 Alan 组已有两个 agent 分别偏：

* **Scientist / Planner Agent**：把意图变成实验计划、调用工具、做 BO 或 protocol 选择
* **Data / Analysis Agent**：处理数据、生成结论、建议下一轮（或做知识库/记忆）

即使实际名字不同，接口仍然通用。

---

## 1) 三个 agent 的明确分工与位置

```
Alan-Agent-A  (Decision / Scientist)
    |
    |  HighLevelPlan (what to do)
    v
Your Agent    (Execution + Recovery)   <-- 你做的灾难恢复在这里
    |
    |  Outcome + Evidence (what happened, auditable)
    v
Alan-Agent-B  (Analysis / Memory)
    |
    |  NextRunSuggestion / Knowledge Update
    v
Alan-Agent-A
```

关键原则：

* **Alan 的 agent 负责“选实验/改策略/决定下一步”**
* **你的 agent 负责“执行并保证安全与可恢复”**
* **你的 agent 输出必须结构化、可回放**，这样 Alan 的 agent 才能 reliable 地消费

---

## 2) 你需要提供的 2 个对外接口（最小可用）

### Interface 1：`execute_plan(plan) -> RunResult`

这是 Alan agent 调你 agent 的主入口。

**Plan 不是一堆 tool calls**，而是“可验证、可解释的执行计划”：

```json
{
  "plan_id": "uuid",
  "workflow_name": "zn_electrodeposition_v3",
  "steps": [
    {"step_id":"s1", "action":"pump.dispense", "params":{"reagent":"ZnSO4","ml":1.0}},
    {"step_id":"s2", "action":"robot.move", "params":{"to":"cell"}},
    {"step_id":"s3", "action":"potentiostat.run", "params":{"profile":"CV", "loops":20}}
  ],
  "constraints": {
    "safety": {"max_temp":130, "max_current":2.0},
    "budgets": {"max_retries":3, "max_total_steps":500}
  },
  "checkpoints": ["after_s1", "after_s3"],
  "criticality": {"s1":"high","s2":"medium","s3":"high"}
}
```

Alan agent 生成这个 plan，你的 agent 执行并处理异常。

---

### Interface 2：`RunResult`（你的 agent 回给 Alan 的结构化回执）

你的输出必须让上层 agent 能做两件事：

1. 决策下一轮（继续/改参数/replicate）
2. 学习与记忆（错误模式、有效恢复策略）

```json
{
  "run_id": "uuid",
  "plan_id": "uuid",
  "status": "success | skip | abort",
  "sample_status": "intact | compromised | destroyed | anomalous",
  "final_state": { "devices": {...}, "hazards": [...], "irreversible_actions": [...] },
  "events_uri": "s3://.../events.jsonl",
  "artifacts": {
    "raw_data_uri": "s3://.../raw/",
    "kpi_uri": "s3://.../kpi.json"
  },
  "failures": [
    {
      "error_type": "device_disconnected",
      "device": "potentiostat",
      "stage": "electrochem",
      "recovery_actions": ["potentiostat.reconnect", "retry"],
      "recovery_outcome": "failed"
    }
  ],
  "metrics": {
    "recovery_steps": 7,
    "time_to_recover_s": 32,
    "safety_violations": 1
  },
  "recommendations": {
    "next_actions": ["replicate", "degrade_protocol"],
    "notes": "suggest longer OCV stabilization"
  }
}
```

Alan 的 analysis agent 可以把这些写入 memory / KB；decision agent 用 `recommendations` 和 `sample_status` 直接做下一轮。

---

## 3) 你需要留出的“可插拔接口点”（和 Alan 两个 agent 对齐）

你现在的系统里已经有这些模块：Classifier / Playbook / GuardedExecutor / Supervisor / EventLogger。结合 Alan 组 agent 的关键是：**在正确的位置插 hook**。

### Hook A：Plan 提供者（上层）

* `PlanProvider` 接口：Alan agent 实现
* 你的 agent 只要求输入满足 schema

```python
class PlanProvider(Protocol):
    def build_plan(self, objective: dict) -> HighLevelPlan: ...
```

你不需要知道 objective 是 BO 还是人类意图。

---

### Hook B：Recovery Advisor（可选，LLM 或 Alan 的 agent）

* 你的 deterministic recovery 仍是 fallback
* Alan 的 agent 可以作为 `RecoveryAdvisor` 提供排序/诊断建议

```python
class RecoveryAdvisor(Protocol):
    def advise(self, state, error_profile, candidates, history) -> Advice: ...
```

关键：`Advice` 只能在 `candidates` 里选，不能发明动作。

---

### Hook C：Run Result Consumer（下游）

* Alan 的 analysis agent 可以订阅你的事件流/结果流

```python
class RunObserver(Protocol):
    def on_event(self, event): ...
    def on_run_complete(self, run_result): ...
```

你只要保证 event schema 稳定。

---

## 4) 如何“把 Alan 的两个 agent 放进你的 repo”而不耦合

你可以做一个极薄的 `integration/` 层，不在核心里写 Alan-specific 逻辑：

```
src/exp_agent/integration/
  contracts/
    plan.schema.json
    run_result.schema.json
    event.schema.json
  adapters/
    alan_decision_agent_client.py     # 可选：HTTP / gRPC client
    alan_analysis_agent_client.py     # 可选
  hooks.py                            # PlanProvider / RecoveryAdvisor / RunObserver interfaces
```

这样 Alan 那边只要实现 client/server，你这边不改 core。

---

## 5) 最小结合方案（你下周就能在组会上跑通）

不要求你知道 Alan 两个 agent 的内部实现，先做“接口占位 + 模拟”。

### Step 1：固定一个 `HighLevelPlan schema`（你来定）

* Alan agent 只要能输出符合 schema 的 JSON
* 你执行并产出 RunResult

### Step 2：把你的 `RunResult` 发给 Alan 的 analysis agent（哪怕先是文件）

* 先最简单：写到 S3 / 本地文件夹
* Alan analysis agent 读取并总结成“next-run suggestion”

### Step 3：把 suggestion 回注入到 Alan decision agent（哪怕先人工）

* 先人为把 suggestion 粘贴进去
* 后面再自动化（webhook / message bus）

这就实现了 **Agent ecosystem 闭环**，哪怕没有任何 LLM 直接干预执行。

---

## 6) 你需要向 Alan 组要的最少信息（不需要问很多）

为了真正对齐那两个 agent，你只要拿到这三件事：

1. Alan decision agent 现在的输入输出格式（哪怕是非正式 JSON）
2. Alan analysis agent 能消费什么（events? kpi? raw data?）
3. 他们偏好的通信方式：文件（S3）/ HTTP / message bus（Kafka/NATS）

拿不到也没关系，你先用你自己的 schema 做出 “adapter layer”，把字段映射一下就行。

---

## 7) 你下一步应该做什么（最短路径）

1. 在你 repo 增加：

* `contracts/plan.schema.json`
* `contracts/run_result.schema.json`
* `integration/hooks.py`

2. `Supervisor.execute_plan(plan)` 作为唯一入口

3. `RunResult` 输出里加入：

* `events_uri`
* `metrics`
* `sample_status`（尤其 `anomalous`）

---

如果你把 Alan 组那两个 agent 的名字/一两句定位贴出来（或者他们 repo 的 README 里怎么描述输入输出），我可以把上面这套接口**映射成更精确的字段与调用链**，并给你一个“组会一页图”展示三者如何组成 ecosystem。
