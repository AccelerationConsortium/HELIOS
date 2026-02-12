下面是对你这份“旧版灾难恢复 agent 文档”的审查：哪些需要修改、哪些需要删、哪些需要补，以及下一步该做什么。基于你发的 `AGENT_DOCUMENTATION.md` 内容。

## 需要修改的点（按优先级）

### 1) “Sense/Monitor” 描述不严谨：把科学异常和执行异常混在一起

文档里把异常检测写成温度过冲、超时、传感器故障、安全违规等（执行/安全异常），但没有明确区分：

* **Safety anomaly**（必须立即终止/降级）
* **Execution anomaly**（设备/通信/动作未达成）
* **Scientific anomaly**（执行正确但结果 outlier，可能是新现象）

**要改：**

* 在“智能错误分类”里增加第三通道：`ScientificAnomaly`（anomalous-but-valid）
* 明确：RecoveryAgent 只管 execution correctness + safety；科学 outlier 进入“quarantine + replicate + notify”通道，而不是 abort/skip。

这直接对应你最近的 concern：outlier 不应该被当 error 杀掉。

---

### 2) “可恢复/不可恢复/不安全”三分法不够：缺少“动作可用候选集”和“诊断阶段”

目前是：Classifier → RecoveryAgent → Decision（retry/skip/degrade/abort）。但**缺两块会导致规则失控**：

* **Diagnostics（只读）阶段**：先做 read-only probes（health/read_state/查询状态）再决策，而不是盲目 retry。
* **Candidate action whitelist（playbook）**：LLM/规则都只能从候选集中选，不能自由发明动作。

**要改：**

* `ErrorProfile` 增加：`diagnostics: [Action]`、`candidates: [ActionTemplate]`、`unsafe: bool`、`recoverable: bool`、`tags`。
* `RecoveryAgent` 流程变为：`classify -> diagnostics -> signature -> choose(decision from candidates)`。

---

### 3) “GuardedExecutor 三层检查”写得对，但缺一个硬安全要求：unsafe preemption

文档里提到 PostCheck 轮询验证、强制安全不变量（如温度始终 <130°C），但没有强调一个必须的语义：

> **在 postcondition polling 过程中，只要触发 unsafe，必须立即抛出 unsafe_violation，而不是等 timeout 再报 postcondition_failed。**

这点你在新版实现里已经在做，但旧文档没有明确“优先级”。

**要改：**

* 在 PostCheck 小节写清楚：`unsafe assertions preempt postconditions`（优先级最高）。

---

### 4) WorkflowManager 的抽象容易误导：把“工作流编排”和“恢复策略”耦合

旧文档把 WorkflowManager 描述得很像一个“大 orchestrator”，容易把 recovery 绑到步骤序号/阶段，从而回到你一开始想避免的 hardcode。

**要改：**

* 明确 WorkflowManager 只负责：**step graph + checkpoint + step criticality**。
* Recovery policy 只依赖：**ErrorProfile + State Features + Step criticality**，不依赖“第几步”。

建议在文档里加一句：policy should be stable under workflow refactors.

---

### 5) 配置文件设计（lab_config / workflow_config）需要统一为“合同（contract）+ policy（playbook）+ tests”

现在的 json 配置把设备、阈值、workflow step 混在一起。这是可以跑，但不利于共享与版本化。

**要改：**

* `contract.yaml`：设备能力/约束（相当于 MCP 的可执行 contract）
* `playbook.yaml`：错误 → 候选策略 → 决策
* `tests.yaml`：输入 state/error → 期望 decision（CI 可跑）
* workflow 仍可用 json/yaml，但 recovery 不应该写在 workflow 里。

---

## 建议删除或降级的表述

### “智能分类/智能决策”的措辞要降级

旧文档大量使用“智能”“像技术员”。在组会/工程评审里，这会被追问：智能在哪里？是否可证明？

**替换成可验证措辞：**

* “event-sourced execution + verifiable postconditions”
* “policy-driven recovery with budgets and terminal invariants”
* “optional LLM advisor constrained by candidates”

---

## 下一步做什么（最值得做的 4 个工程里程碑）

### Milestone 1：把 recovery 从“代码逻辑”固化成 playbook + tests（skill 化）

产出：

* `contract.yaml`（动作、参数 schema、安全包络、postconditions）
* `playbook.yaml`（ErrorProfile + features → Decision）
* `tests.yaml`（overshoot、timeout、sensor_fail、comm_drop 等 10 个场景）

验收：

* workflow 重排不改 playbook（或只改极少数规则）
* CI 跑 tests 全绿

---

### Milestone 2：加入 ScientificAnomaly 通道（避免 discovery 被 abort/skip）

产出：

* `sample_status` 扩展为：`intact/compromised/destroyed/anomalous`
* “执行正确但 KPI 异常” → 标记 anomalous + quarantine + replicate suggestion
* 事件日志里增加 `ScientificAnomalyDetected`

验收：

* unsafe 仍然第一优先级
* scientific anomaly 不触发 abort（除非同时 unsafe）

---

### Milestone 3：把你的电池 recovery 逻辑抽象成 state features + policy（你最近想做的那件事）

产出：

* 一套 battery 的 `features`（例如：potentiostat running? electrode held? reagent_added?）
* 一份 battery `playbook.yaml`
* 对应模拟/回放数据的 replay + metrics

验收：

* 至少 5 类真实失败（串口超时、连接断开、泵堵塞、阀不响应、机械臂撞限位）能走完“诊断→策略→恢复/安全终止”
* 输出 metrics：recovery steps、time-to-recover、abort rate

---

### Milestone 4：LLM Advisor 只做“候选排序/诊断建议”（可选，放最后）

产出：

* `LLMAdvisor`（结构化输入输出）
* `DecisionArbiter`（严格拒绝非候选动作；unsafe 下只允许 abort/degrade candidates）
* 记录 `advice_accepted/rejected`

验收：

* LLM 断网系统仍能运行
* metrics 在 repeated failures 上优于纯规则（更少 retry、更快 degrade/abort）

---

## 你这份文档最该立刻改的 6 行“总纲”

建议你把总结段落改成类似：

* 系统目标：**安全与执行正确性**
* Recovery = **policy-driven**，与 workflow 解耦
* 事件日志 = **事实源**，支持 replay/metrics
* Scientific anomaly = **单独通道**（anomalous-but-valid），支持 discovery
* LLM = **optional advisor**（只做诊断与候选排序，不触碰硬件）

我的想法：
新版 policy pseudo-code”（直接替换掉现有 10 条）

```pseudo
function decide_recovery(state, error, last_action) -> Decision:
  profile = classify(error)                 # unsafe/recoverable/default_strategy
  mode = signature(error.telemetry_window)  # drift/oscillation/stall/noisy/stable
  stage = state.stage
  target = last_action.params.target_c if exists

  # 0) unsafe preemption
  if profile.unsafe:
      # Always go to a safe envelope first
      actions = [cool_down()]
      if profile.recoverable and mode in {drift, noisy} and stage != "cleanup":
          degraded = compute_degraded_target(target, mode)
          actions += [set_temperature(degraded), wait(stabilize_time(mode))]
          return Decision("DEGRADE", actions, mark_sample_status=maybe_compromised(stage))
      return Decision("ABORT", actions, mark_sample_status="destroyed")

  # 1) non-recoverable
  if not profile.recoverable:
      return Decision("ABORT", [cool_down()], mark_sample_status="compromised")

  # 2) recoverable: choose by mode + retry_count with backoff
  r = state.retry_count_by_error_type[error.type]

  if error.type in {timeout, communication_error}:
      return Decision("RETRY", [wait(backoff(r)), retry_original()], mark_sample_status="intact")

  if error.type == postcondition_failed:
      if mode == stall:
          return Decision("ABORT", [cool_down()], mark_sample_status="compromised")
      if r == 0:
          return Decision("RETRY", [retry_original()], mark_sample_status="intact")
      if r == 1:
          return Decision("RETRY", [wait(2), retry_original()], mark_sample_status="intact")
      # repeated: degrade based on target
      degraded = compute_degraded_target(target, mode)
      return Decision("DEGRADE", [cool_down(), set_temperature(degraded)], mark_sample_status="compromised")

  # fallback
  return Decision("ABORT", [cool_down()], mark_sample_status="compromised")


function compute_degraded_target(target, mode):
  if target is None: return 110
  delta = (10 if mode in {drift, noisy} else 15)
  return max(ambient_temp(), target - delta)

function stabilize_time(mode):
  if mode == oscillation: return 10
  if mode == drift: return 5
  return 2

function backoff(retry_count):
  # 0->0s, 1->2s, 2->5s, 3->10s capped
  return min([0,2,5,10][min(retry_count,3)], 10)
```


### Step 1：把 10 条规则迁移成 “policy 表达层”

即使你暂时不做 YAML，也先让它们集中在一个文件里（例如 `recovery/policy.py`），不要散落在 RecoveryAgent/Classifier。

交付标准：所有 recovery 决策都由 `decide_recovery()` 单点产出。

### Step 2：把 signature 扩展到 4 类，并把阈值做成 config

* slope threshold、stall eps、oscillation amplitude threshold 全部进 config
* 输出 mode + confidence + features

### Step 3：把 retry 机制升级为 backoff + degrade escalation

* 同一 error 重复出现，策略必须升级，不允许纯 retry 打满预算才 abort

### Step 4（可选）：LLM advisor 接入点只放在“repeated failures”

输入：profile + mode + telemetry_window + candidates
输出：在 candidates 里排序/选策略（不发明动作）

---

## 5) 你提的 “让 LLM 学习决策模式并泛化” 的正确做法

不是把 10 条 if-else 喂给 LLM 让它模仿，而是：

* 你用 policy 生成 **候选集 candidates**
* LLM 在 candidates 里选一个，并给 rationale
* Arbiter/GuardedExecutor 兜底拒绝不安全/不合法建议
* 通过 logs 统计：LLM 是否减少 recovery steps / time_to_recover

---
至少需要完成，一份 `playbook.yaml`（包含 mode/stage/retry_count 条件）+ 一份 `signature.yaml`（阈值配置）+ 10 个 `tests.yaml` 场景（overshoot、sensor_fail、timeout、postcondition_failed + drift/stall/oscillation），这样你把“从硬编码到 policy-driven”一次性完成并可 CI 验收。
