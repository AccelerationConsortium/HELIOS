# RecoveryAgent 快速开始指南

## ✅ 已完成集成

RecoveryAgent已经完全集成到OTbot的所有层级，无需额外配置即可使用。

---

## 🚀 立即使用

### 1. 启动OTbot服务

```bash
cd /Users/sissifeng/OTbot
python3 app/main.py
```

服务启动后，访问: http://localhost:8000/static/lab.html

### 2. 开始Campaign

在Lab Agent UI中：
1. 输入实验描述（例如："Serial dilution with 96-well plate"）
2. 点击发送启动campaign
3. RecoveryAgent自动在后台工作

---

## 🔍 Recovery功能自动生效场景

### ✅ 自动Retry (无需人工干预)
**场景**: OT-2连接超时
```
❌ 执行失败: TimeoutError
  ↓
🔄 RecoveryAgent决策: retry (2s延迟)
  ↓
✅ 第2次尝试成功
  ↓
前端显示: ✅ Success after 1 retry
```

### ✅ 降级运行 (继续campaign)
**场景**: 温度传感器异常
```
❌ 执行失败: sensor_fail
  ↓
⚠️  RecoveryAgent决策: degrade
  ↓
✅ 使用备用传感器继续
  ↓
前端显示: ⚠️ Recovery: degrade (degraded mode)
```

### ✅ 跳过Candidate (继续其他候选)
**场景**: 某个candidate参数导致执行失败
```
❌ Candidate #2 执行失败: RuntimeError
  ↓
⏭️  RecoveryAgent决策: skip
  ↓
✅ 跳过#2，继续执行#3
  ↓
前端显示: ⏭️ Recovery: skip
```

### 🚨 化学安全Abort (强制停止)
**场景**: 检测到液体泄漏
```
❌ 检测到: spill_detected
  ↓
🚨 RecoveryAgent识别化学安全事件
  ↓
🛡️  SafetyAgent否决权启动
  ↓
⛔ 强制abort campaign
  ↓
前端显示: 🚨 CHEMICAL SAFETY EVENT: spill_detected
```

---

## 📊 前端UI显示

### Recovery事件样式

**正常Retry**:
```
🔄 Recovery: retry (attempt 1)
ℹ️  timeout → 🔄 Recovery: retry
```

**成功恢复**:
```
✅ Success after 2 retries
```

**降级运行**:
```
⚠️  Recovery: degrade
```

**跳过Candidate**:
```
⏭️  Recovery: skip
```

**化学安全警报**:
```
🚨 CHEMICAL SAFETY EVENT: spill_detected
SafetyAgent veto active
```

**超过重试次数**:
```
❌ Failed after 3 retries
```

### 严重程度图标
- 🚨 **高严重程度** (high): 化学安全、safety violation
- ⚠️  **中等严重程度** (medium): sensor fail, actuator jam
- ℹ️  **低严重程度** (low): timeout, drift

---

## 🧪 测试验证

### 运行测试
```bash
# RecoveryAgent单元测试
python3 -m pytest tests/test_recovery_agent.py -v

# 集成测试
python3 -m pytest tests/test_recovery_integration.py -v

# Orchestrator测试
python3 -m pytest tests/test_orchestrator.py -v

# 全部测试
python3 -m pytest tests/test_recovery*.py tests/test_orchestrator.py -v
```

### 预期结果
```
57 passed in 0.38s ✅
```

### Demo演示
```bash
PYTHONPATH=/Users/sissifeng/OTbot python3 examples/recovery_agent_demo.py
```

---

## 📖 详细文档

### 完整集成指南
📄 [`docs/RECOVERY_AGENT_INTEGRATION.md`](RECOVERY_AGENT_INTEGRATION.md)
- RecoveryAgent详细API
- 自定义recovery策略
- SafetyPacket集成
- LLM advisor配置

### 集成完成报告
📄 [`docs/RECOVERY_INTEGRATION_COMPLETE.md`](RECOVERY_INTEGRATION_COMPLETE.md)
- 完整架构图
- 文件清单
- 性能指标
- 未来增强计划

### 集成总结
📄 [`docs/RECOVERY_AGENT_SUMMARY.md`](RECOVERY_AGENT_SUMMARY.md)
- 初步集成总结
- 下一步建议 (已完成)

---

## ⚙️ 配置选项

### 修改Retry参数

在 `app/agents/orchestrator.py` 的 `_execute_candidate_with_recovery()` 方法中：

```python
max_retries = 3  # 修改最大重试次数 (默认3)
```

### 修改Retry延迟

RecoveryAgent自动决定延迟时间（通常2-5秒），基于：
- 错误类型
- 重试次数
- 历史模式

要自定义，修改 `recovery-agent/src/exp_agent/recovery/policy.py`

### 化学安全阈值

在 `app/services/error_mapping.py` 中修改：

```python
def should_emit_chemical_safety_alert(error_type, telemetry):
    temp = telemetry.get("temperature")
    if temp and temp > 80.0:  # 修改温度阈值
        return True

    pressure = telemetry.get("pressure")
    if pressure and pressure > 2.0:  # 修改压力阈值
        return True
```

---

## 🐛 故障排查

### Recovery不工作？

1. **检查RecoveryAgent状态**:
```python
from app.agents import OrchestratorAgent
orchestrator = OrchestratorAgent()
print(orchestrator.recovery._available)  # 应该是True
```

2. **查看日志**:
```bash
# 启动时带日志
python3 app/main.py 2>&1 | grep -i recovery
```

3. **测试Recovery**:
```bash
python3 -m pytest tests/test_recovery_agent.py::test_recovery_agent_basic -v
```

### 化学安全事件未触发？

检查错误映射:
```python
from app.services.error_mapping import should_emit_chemical_safety_alert

# 测试
result = should_emit_chemical_safety_alert("spill_detected", {})
print(result)  # 应该是True
```

### 前端未显示Recovery事件？

1. 检查浏览器console (F12)
2. 查看SSE连接状态
3. 检查备份轮询是否工作

---

## 📊 监控Recovery指标

### 查看Campaign结果

完成后，campaign结果包含recovery metrics:
- `total_errors`: 总错误数
- `retry_count`: 重试次数
- `abort_count`: Abort次数
- `skip_count`: Skip次数
- `chemical_safety_events`: 化学安全事件数

### 实时监控

在campaign运行时，frontend实时显示：
- Recovery决策
- Retry尝试次数
- 成功/失败状态

---

## 💡 最佳实践

### 1. 让RecoveryAgent自动工作
- ✅ 不需要手动配置
- ✅ 自动处理大多数错误
- ✅ 只在必要时人工干预

### 2. 关注化学安全警报
- 🚨 化学安全事件需要立即人工检查
- 🚨 不要忽略SafetyAgent否决
- 🚨 检查设备状态后再继续

### 3. 监控Retry模式
- 如果某个错误频繁retry → 检查硬件
- 如果所有retries都失败 → 可能需要维护
- 如果经常degrade → 考虑校准传感器

### 4. 定期审查Recovery日志
- 分析最常见的错误类型
- 优化容易失败的操作
- 更新recovery策略

---

## 🎯 下一步

RecoveryAgent已经完全集成并可以使用！

**立即开始**:
1. 启动OTbot: `python3 app/main.py`
2. 打开Lab Agent UI: http://localhost:8000/static/lab.html
3. 启动campaign，RecoveryAgent自动工作

**需要帮助?**
- 查看完整文档: `docs/RECOVERY_AGENT_INTEGRATION.md`
- 运行demo: `python3 examples/recovery_agent_demo.py`
- 运行测试: `pytest tests/test_recovery*.py -v`

---

**状态**: ✅ Production Ready
**测试**: 57 passed
**版本**: v1.0 (2026-02-11)
