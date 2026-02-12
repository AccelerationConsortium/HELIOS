# Exp-Agent Test Demo 视频讲解稿

> 视频文件: `demo/recordings/exp-agent-test-demo.mp4` (约 56 秒)
>
> 本文档按视频时间线逐段讲解：每段对应什么测试、注入了什么故障、agent 做了什么决策、映射到架构的哪个模块。

---

## 0:00 – 0:02  开场标题

**画面**: 标题屏 "Exp-Agent: Recovery-Aware Execution Agent — Test Suite Demo — 29 Tests"，列出 9 项测试覆盖。

**讲解**:
> 这是我们 Exp-Agent 的完整测试套件演示。29 个测试覆盖了 agent 的三层决策架构：底层的错误分类与签名分析、中层的恢复策略决策、顶层的工作流调度与故障降级。

---

## PART 1: 核心策略与恢复测试 (21 个测试)

### 0:02 – 0:03  Guardrails 安全检查 (1 个测试)

**画面**: `[1/3] Running Guardrails Tests...` → `test_guardrails_safety_check PASSED`

**讲解**:
> 第一个测试验证 GuardedExecutor 的安全门。它是执行层的"保险丝"——在每次 action 执行前检查 precondition 和 safety constraint，执行后用 PostCheck 轮询验证 postcondition。如果任何安全约束被违反，执行会被立即拦截。
>
> **对应模块**: `executor/guarded_executor.py` — Pre-check / Safety-check / Post-check 三阶段守卫

### 0:04 – 0:05  Recovery Policy 测试 (15 个测试)

**画面**: 15 个测试快速滚过，全部 PASSED。

这 15 个测试分三组：

#### 签名分析 (Signature Analysis) — 5 个测试 (~0:05)

| 测试 | 注入的故障模式 | 验证内容 |
|------|---------------|----------|
| `test_drift_detection` | 构造温度持续上升的 history（每步 +2°C） | agent 分析历史遥测数据，识别出 **drift（漂移）** 签名 |
| `test_stall_detection` | 构造温度完全不变的 history | 识别出 **stall（停滞）** 签名 |
| `test_oscillation_detection` | 构造温度在两个值间来回跳动的 history | 识别出 **oscillation（振荡）** 签名 |
| `test_stable_detection` | 构造温度保持在目标附近的 history | 识别出 **stable（稳定）** 签名 |
| `test_insufficient_history` | 只给 2 个 history 点 | 返回 **unknown** — 数据不足以判断 |

**讲解**:
> 这组测试验证 agent 的"诊断能力"。当硬件报错时，agent 不是简单地看错误类型，而是分析最近的遥测历史——温度是在漂移、停滞、还是振荡？不同的签名对应不同的恢复策略。这就像医生不只看症状，还要看病程。
>
> **对应模块**: `recovery/policy.py → analyze_signature()` — 基于滑动窗口的时序签名分析

#### 错误分类 (Error Classification) — 3 个测试 (~0:05)

| 测试 | 注入的故障 | agent 的分类结果 |
|------|-----------|-----------------|
| `test_overshoot_classification` | `HardwareError(type="overshoot")` 温度超调 | unsafe=True, recoverable=True — 危险但可恢复 |
| `test_sensor_fail_classification` | `HardwareError(type="sensor_fail")` 传感器故障 | unsafe=True, recoverable=False — 危险且不可恢复 |
| `test_timeout_classification` | `HardwareError(type="timeout")` 操作超时 | unsafe=False, recoverable=True — 安全可重试 |

**讲解**:
> 这组验证 agent 的"分诊能力"。每个硬件错误进来，先判断两个维度：是否 unsafe（是否有安全风险）、是否 recoverable（是否有恢复可能）。这决定了后续走 retry、degrade、还是 abort。
>
> **对应模块**: `recovery/classifier.py → classify_error()` — 错误分类器

#### 恢复决策 (Recovery Decisions) — 7 个测试 (~0:05)

| 测试 | 注入场景 | agent 的决策 | 对应策略 |
|------|---------|-------------|---------|
| `test_overshoot_with_drift_degrade` | overshoot + drift 签名 + 有 target 上下文 | **DEGRADE** — 降低目标温度 10°C | 可恢复的漂移 → 降级运行 |
| `test_sensor_fail_abort` | sensor_fail（不可恢复） | **ABORT** — 立即安全停机 | 不可恢复 → 终止 |
| `test_timeout_first_retry` | 首次 timeout | **RETRY** — 等 2 秒后重试 | 安全可恢复 → 重试 |
| `test_timeout_with_backoff` | 第 2 次 timeout | **RETRY** with backoff — 等 4 秒 | 指数退避 |
| `test_postcondition_stall_abort` | postcondition 失败 + stall 签名 | **ABORT** — 设备无响应 | 停滞 → 无法恢复 |
| `test_postcondition_escalation_to_degrade` | postcondition 失败 + drift 签名 | **DEGRADE** — 降级 | 漂移 → 降级运行 |
| `test_full_escalation_sequence` | 连续 3 次 timeout | RETRY → RETRY(backoff) → **ABORT** | 重试预算耗尽 → 终止 |

**讲解**:
> 这是最核心的一组。每个测试模拟一种特定的故障场景，验证 agent 走完整的 classify → signature → decide 管线后做出正确的决策。关键设计点：
> - **RETRY** 带指数退避（2s → 4s），不是盲目重试
> - **DEGRADE** 只在"漂移"签名 + 有可调目标时触发，会降低目标温度继续运行
> - **ABORT** 是最后手段，会触发安全停机
> - 重试有预算上限，耗尽后自动升级到 abort
>
> **对应模块**: `recovery/policy.py → decide_recovery()` — 核心决策函数；`recovery/recovery_agent.py` — 管线编排

### 0:06  Recovery Agent 集成测试 (5 个测试)

**画面**: `[3/3] Running Recovery Agent Tests...` → 5 passed

**讲解**:
> 这组验证 RecoveryAgent 作为编排层，把 classifier、signature analyzer、policy 正确串联。包括验证 overshoot 在有/无上下文时的不同决策（有上下文 → degrade，无上下文 → abort），以及重试计数器的累加和签名分析的端到端集成。
>
> **对应模块**: `recovery/recovery_agent.py` — 恢复管线编排器

---

## PART 2: 工作流调度测试 (8 个测试)

### 0:08 – 0:14  Happy Path — 全步骤正常执行

**画面**: `test_happy_path_all_steps_ok` — WorkflowSupervisor 执行 4 步计划：
```
[1/4] step_id=setup    stage=setup      → ✓ STEP OK
[2/4] step_id=heat     stage=heating    → ✓ STEP OK  (PostCheck 轮询 ~5s)
[3/4] step_id=snapshot  stage=diagnostics → ✓ STEP OK
[4/4] step_id=cooldown  stage=cooldown   → ✓ STEP OK
PLAN COMPLETE — ok=4 skipped=0 degraded=0 aborted=0
```

**讲解**:
> 第一个 workflow 测试是 happy path：无故障注入，验证 WorkflowSupervisor 的 execute_plan() 按顺序走 4 个步骤。注意几个关键点：
> - 每步都显示 `step_id` 和 `stage`，这是工作流游标追踪
> - `set_temperature` 步骤有 ~5 秒的 PostCheck 轮询等待（模拟加热过程中的温度渐进）
> - 最后的 SUMMARY 统计了各类 outcome
>
> **对应模块**: `orchestrator/workflow_supervisor.py → execute_plan()` — 工作流主循环

### 0:15 – 0:21  Step IDs 和 Stages 验证

**画面**: `test_step_ids_and_stages_in_output` — 同样执行 4 步计划，验证每个 StepResult 都包含 step_id 和 stage 字段。

**讲解**:
> 这个测试确保工作流输出的结构化数据完整。每个步骤的执行结果都必须携带 step_id 和 stage，这是后续做审计追踪和故障定位的基础。

### 0:21 – 0:23  ⭐ SKIP 决策 — optional 步骤失败 → 跳过

**画面**: `test_optional_step_fail_triggers_skip` — 这是关键场景：
```
[1/3] step_id=setup → ✓ STEP OK
[2/3] step_id=optional_check  criticality=optional  on_failure=skip
      ✗ STEP FAILED  error=postcondition_failed
      DECISION: retry — Postcondition failed. Retry with 2s wait.
      CURSOR: retry (1/2)
[2/3] step_id=optional_check (重试)
      ✗ STEP FAILED  error=postcondition_failed
      Signature: stall (confidence=0.90)
      DECISION: abort — Postcondition failed with stall signature.
      CURSOR: skip (on_failure override for optional step)  ← 关键！
[3/3] step_id=final → ✓ STEP OK
PLAN COMPLETE — ok=2 skipped=1
  ⊘ optional_check (diagnostics) [skip: Optional step abort → skip override]
```

**讲解**:
> **这是整个 demo 最值得关注的场景。** 我们注入了一个不可能满足的 postcondition（要求温度达到 999°C），观察 agent 的决策链：
>
> 1. **第一次失败**: postcondition 验证失败 → 分类为 recoverable → Policy 决定 **RETRY**，等 2 秒重试
> 2. **第二次失败**: 再次失败 → 分析签名发现是 **stall**（温度完全不动）→ Policy 决定 **ABORT**
> 3. **关键转折**: 但 WorkflowSupervisor 检查了步骤的 `criticality=optional` 和 `on_failure=skip`，**把 abort 覆盖为 skip**
> 4. **工作流继续**: 游标前进到步骤 3，最终计划成功完成
>
> 这展示了两层决策的配合：Recovery Policy 根据硬件状态做"技术判断"，WorkflowSupervisor 根据步骤重要性做"业务判断"。一个 optional 的诊断步骤失败不应该终止整个实验。
>
> **对应模块**:
> - `recovery/policy.py` — 技术层决策（retry → abort）
> - `orchestrator/workflow_supervisor.py` 的游标逻辑 — 业务层覆盖（abort → skip）
> - `core/types.py → PlanStep.criticality / on_failure` — 步骤语义定义

### 0:23 – 0:25  ⭐ ABORT 决策 — critical 步骤失败 → 终止

**画面**: `test_critical_step_fail_triggers_abort`:
```
[1/3] step_id=setup → ✓ STEP OK
[2/3] step_id=critical_heat  criticality=critical  on_failure=abort
      ✗ STEP FAILED  error=postcondition_failed
      DECISION: retry
      CURSOR: retry budget exhausted (1/0)  ← max_retries=0，立即耗尽
      CURSOR: abort (on_failure fallback)
      SHUTDOWN: safe shutdown sequence
```

**讲解**:
> 同样是 postcondition 失败，但这次步骤标记为 `criticality=critical`、`on_failure=abort`、`max_retries=0`。agent 的行为完全不同：
> - Policy 想 retry，但重试预算为 0，立即耗尽
> - 回退到 `on_failure=abort` → 触发安全停机
> - 步骤 3（final）不会执行 — 计划直接终止
>
> 对比上一个测试：同一种错误，因为步骤的 **criticality 不同**，产生了完全不同的结果。这就是"业务语义驱动决策"的核心设计。

### 0:25  PlanPatch 结构验证

**画面**: `test_plan_patch_structure PASSED`（瞬间通过，无详细输出）

**讲解**:
> 验证 PlanPatch 数据结构的字段完整性：`overrides`（参数覆盖）、`relaxations`（postcondition 放宽）、`notes`（人类可读说明）、`original_target`（原始目标）、`degraded_target`（降级目标）。

### 0:25 – 0:28  ⭐ DEGRADE 级联 — overshoot → 降级 → 下游 patch

**画面**: `test_degrade_updates_downstream_postconditions`:
```
[1/4] step_id=setup → ✓ STEP OK
[2/4] step_id=heat  stage=heating
      ✗ STEP FAILED  error=safety_violation
      DECISION: abort — Unsafe condition (safety_violation)
      SHUTDOWN: safe shutdown sequence
```

**讲解**:
> 这个测试使用 `fault_mode=overshoot` 的 SimHeater。overshoot 模式下加热器会超调 20°C/步，触发 safety_violation。在这次运行中，由于历史数据不足（签名分析返回 unknown），Policy 直接 abort 了。
>
> 但测试的核心验证逻辑在条件分支里：**如果** degrade 发生了，会验证 PlanPatch 的级联效果 —— downstream 的 `hold` 步骤的 postcondition 从 `~= 120.0` 被 patch 为 `~= <degraded_target>`，不再引用旧目标。

### 0:28  ⭐ PlanPatch 应用验证

**画面**: `test_patch_apply_changes_action_params`:
```
PATCH APPLIED: hold.temperature: 120.0 → 110.0
PATCH APPLIED: hold postconditions relaxed
```

**讲解**:
> 这个测试直接验证 `_apply_patches()` 方法。手动创建一个 PlanPatch（原目标 120°C → 降级目标 110°C），然后将它应用到一个引用 120°C 的 downstream step：
> - `params.temperature`: 120.0 → **110.0**（参数覆盖）
> - `postconditions`: `"~= 120.0"` → **`"~= 110.0"`**（约束放宽）
>
> 这确保降级决策不是只改当前步骤，而是**级联到所有下游步骤**——否则后续步骤会用旧的标准去校验新的运行状态，导致不必要的失败。
>
> **对应模块**: `orchestrator/workflow_supervisor.py → _apply_patches() / _build_patch()` — PlanPatch 构建与应用

### 0:28 – 0:29  重试预算耗尽 → 回退到 on_failure

**画面**: `test_retry_budget_exhausted_optional_skips`:
```
[1/2] step_id=optional_bad  criticality=optional  on_failure=skip
      ✗ STEP FAILED  error=postcondition_failed
      DECISION: retry
      CURSOR: retry budget exhausted (1/0)
      CURSOR: skip (on_failure fallback)
[2/2] step_id=end → ✓ STEP OK
PLAN COMPLETE — ok=1 skipped=1
  ⊘ optional_bad [skip: Retry budget exhausted, on_failure=skip]
```

**讲解**:
> 最后一个 workflow 测试：optional 步骤设置 `max_retries=0`。Policy 决定 retry，但预算立即耗尽，回退到步骤的 `on_failure=skip`。工作流继续，计划成功。这验证了"重试预算"作为安全阀的设计——即使 Policy 想一直重试，预算机制也会强制终止并走 fallback。

### 0:29  8 passed in 21.44s

**画面**: `8 passed` — 全部 Workflow Supervisor 测试通过。

---

## PART 3: 全套件总结 (0:31 – 0:56)

### 0:31 – 0:53  完整测试套件运行

**画面**: 29 个测试名依次滚过，全部绿色 PASSED。

**讲解**:
> 最后一次完整运行全部 29 个测试作为确认。注意 workflow 测试（75%-100%）有明显的执行间隔——那是真实的 PostCheck 轮询等待和模拟加热时间。

### 0:53 – 0:56  总结画面

**画面**:
```
All 29 tests passed!

Decision types tested:
  RETRY   — timeout with backoff, retry budget exhaustion
  SKIP    — optional step failure → skip, continue workflow
  DEGRADE — overshoot with drift → lower target, patch downstream
  ABORT   — sensor failure, safety violation → safe shutdown

Workflow features tested:
  step_id / stage cursor tracking
  PlanPatch cascading (overrides + relaxations)
  Criticality semantics (critical vs optional)
  on_failure fallback (abort / skip)
```

**讲解**:
> 总结了 agent 覆盖的 4 种决策类型和 4 项工作流特性。核心设计思想是**分层决策**：
>
> 1. **最底层** — GuardedExecutor：执行前安全检查，执行后 postcondition 验证
> 2. **中间层** — RecoveryAgent：classify → signature → decide 三阶段管线
> 3. **最顶层** — WorkflowSupervisor：基于步骤 criticality/on_failure 的业务覆盖，加上 PlanPatch 级联
>
> 每一层都可以独立测试，但真正的智能体现在层间协作：Recovery Policy 说 abort，WorkflowSupervisor 说"但这步是 optional 的，skip 就好"。

---

## 架构模块速查

| 视频时间 | 测试场景 | 注入的故障 | Agent 决策 | 对应代码 |
|---------|---------|-----------|-----------|---------|
| 0:03 | 安全门检查 | (正常执行) | Pre/Safety/Post-check 通过 | `executor/guarded_executor.py` |
| 0:05 | 签名分析 | 构造的温度历史 | drift/stall/oscillation/stable/unknown | `recovery/policy.py → analyze_signature()` |
| 0:05 | 错误分类 | overshoot/sensor_fail/timeout | unsafe+recoverable 分类 | `recovery/classifier.py` |
| 0:05 | 恢复决策 | 各类故障组合 | RETRY/SKIP/DEGRADE/ABORT | `recovery/policy.py → decide_recovery()` |
| 0:08-14 | Happy path | 无故障 | 4 步全部 OK | `orchestrator/workflow_supervisor.py` |
| 0:21-23 | **SKIP 覆盖** | 不可能的 postcondition | retry→abort→**skip override** | `workflow_supervisor.py` 游标逻辑 |
| 0:23-25 | **ABORT 终止** | 不可能的 postcondition | retry budget exhausted→abort | `workflow_supervisor.py` 预算机制 |
| 0:25-28 | **DEGRADE 级联** | overshoot → safety_violation | abort (或 degrade + patch 下游) | `workflow_supervisor.py → _build_patch()` |
| 0:28 | **PlanPatch 应用** | 手动注入 patch | 120°C→110°C 参数+约束更新 | `workflow_supervisor.py → _apply_patches()` |
| 0:28-29 | 重试预算 | 不可能的 postcondition | retry→budget exhausted→skip | `workflow_supervisor.py` 预算+fallback |
