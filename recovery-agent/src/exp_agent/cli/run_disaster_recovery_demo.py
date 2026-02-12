#!/usr/bin/env python3
"""
灾难恢复演示：高温反应器热失控场景

展示agent如何综合运用：
1. 设备监控 (Rust TelemetryBuffer)
2. 故障检测与分类 (Recovery Policy)
3. CLI工具调用 (紧急停机、诊断)
4. 本地脚本执行 (数据备份、日志收集)
5. 外部API调用 (LIMS上报、通知系统)

场景：反应器冷却系统故障导致温度失控
"""
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Core types
from exp_agent.core.types import (
    DeviceState, HardwareError, Action, Decision,
    ExecutionState, PlanStep
)

# Recovery system
from exp_agent.recovery.policy import (
    decide_recovery, analyze_signature, RecoveryConfig
)

# External executors
from exp_agent.external.integration import (
    ExternalDevice, ExternalAction,
    create_cli_device, create_script_device, create_api_device,
    create_hybrid_device
)
from exp_agent.external.cli import CLIExecutor, CLIAction, run_command
from exp_agent.external.script import run_python, run_shell
from exp_agent.external.api import APIExecutor, APIAction, HTTPMethod

# Try to import Rust telemetry buffer
try:
    from exp_agent_core import TelemetryBuffer, TelemetryPoint, TelemetryStats
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("⚠️  Rust module not available, using Python fallback")


# ============================================================================
# 模拟设备和环境
# ============================================================================

class SimulatedReactor:
    """模拟的高温反应器"""

    def __init__(self):
        self.name = "reactor_01"
        self.temperature = 150.0  # 当前温度
        self.setpoint = 150.0     # 目标温度
        self.heater_power = 0.6   # 加热功率 0-1
        self.cooling_flow = 1.0   # 冷却水流量 0-1
        self.cooling_fault = False  # 冷却故障标志
        self.emergency_stop = False

        # Telemetry buffer (Rust or Python)
        if HAS_RUST:
            self.telemetry = TelemetryBuffer(1000)
        else:
            self.telemetry = []

    def inject_fault(self):
        """注入冷却系统故障"""
        self.cooling_fault = True
        self.cooling_flow = 0.1  # 流量降至10%
        print("🔴 [FAULT] 冷却水流量传感器故障，流量降至10%")

    def tick(self, dt: float = 1.0):
        """更新物理状态"""
        if self.emergency_stop:
            # 紧急停机：快速降温
            self.heater_power = 0
            self.temperature -= 2.0 * dt
            self.temperature = max(25.0, self.temperature)
        else:
            # 正常物理模拟
            heat_input = self.heater_power * 5.0 * dt
            heat_loss = self.cooling_flow * 3.0 * dt + 0.5 * dt
            self.temperature += heat_input - heat_loss

        # Record telemetry
        timestamp = time.time()
        if HAS_RUST:
            self.telemetry.push(TelemetryPoint(
                timestamp, self.name, "temperature", self.temperature
            ))
            self.telemetry.push(TelemetryPoint(
                timestamp, self.name, "cooling_flow", self.cooling_flow
            ))
        else:
            self.telemetry.append({
                "timestamp": timestamp,
                "temperature": self.temperature,
                "cooling_flow": self.cooling_flow
            })

    def read_state(self) -> DeviceState:
        """读取当前状态"""
        status = "error" if self.cooling_fault else "running"
        return DeviceState(
            name=self.name,
            status=status,
            telemetry={
                "temperature": round(self.temperature, 2),
                "setpoint": self.setpoint,
                "heater_power": self.heater_power,
                "cooling_flow": self.cooling_flow,
                "cooling_fault": self.cooling_fault,
                "emergency_stop": self.emergency_stop,
            }
        )

    def get_history(self) -> List[DeviceState]:
        """获取历史状态用于签名分析"""
        history = []
        if HAS_RUST:
            points = self.telemetry.get_by_device_metric(self.name, "temperature")
            for p in points[-20:]:  # 最近20个点
                history.append(DeviceState(
                    name=self.name,
                    status="running",
                    telemetry={"temperature": p.value}
                ))
        else:
            for entry in self.telemetry[-20:]:
                history.append(DeviceState(
                    name=self.name,
                    status="running",
                    telemetry={"temperature": entry["temperature"]}
                ))
        return history


# ============================================================================
# 外部系统模拟
# ============================================================================

class MockExternalSystems:
    """模拟外部系统响应"""

    @staticmethod
    async def emergency_shutdown(device: str) -> Dict[str, Any]:
        """模拟紧急停机CLI命令"""
        print(f"    🛑 [CLI] labctl emergency-stop --device {device}")
        await asyncio.sleep(0.5)  # 模拟执行时间
        return {"status": "stopped", "device": device}

    @staticmethod
    async def run_diagnostics(device: str) -> Dict[str, Any]:
        """模拟诊断脚本"""
        print(f"    🔍 [SCRIPT] diagnose.py --device {device}")
        await asyncio.sleep(0.3)
        return {
            "device": device,
            "diagnosis": "cooling_pump_failure",
            "confidence": 0.92,
            "recommended_action": "replace_pump"
        }

    @staticmethod
    async def backup_experiment_data(experiment_id: str) -> Dict[str, Any]:
        """模拟数据备份脚本"""
        print(f"    💾 [SCRIPT] backup_data.py --exp {experiment_id}")
        await asyncio.sleep(0.2)
        return {
            "experiment_id": experiment_id,
            "backup_path": f"/backups/{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz",
            "size_mb": 156.7
        }

    @staticmethod
    async def report_to_lims(incident: Dict[str, Any]) -> Dict[str, Any]:
        """模拟LIMS API调用"""
        print(f"    📡 [API] POST https://lims.lab.com/api/v1/incidents")
        await asyncio.sleep(0.2)
        return {
            "incident_id": f"INC-{int(time.time())}",
            "status": "recorded",
            "assigned_to": "on_call_engineer"
        }

    @staticmethod
    async def send_alert(message: str, severity: str) -> Dict[str, Any]:
        """模拟告警通知API"""
        print(f"    📱 [API] POST https://alerts.lab.com/notify")
        await asyncio.sleep(0.1)
        return {"delivered": True, "channels": ["slack", "email", "sms"]}


# ============================================================================
# 灾难恢复协调器
# ============================================================================

class DisasterRecoveryCoordinator:
    """灾难恢复协调器 - 整合所有恢复能力"""

    def __init__(self, reactor: SimulatedReactor):
        self.reactor = reactor
        self.external = MockExternalSystems()
        self.recovery_config = RecoveryConfig()
        self.retry_counts: Dict[str, int] = {}
        self.incident_log: List[Dict[str, Any]] = []

        # 安全阈值
        self.TEMP_WARNING = 160.0
        self.TEMP_CRITICAL = 175.0
        self.TEMP_EMERGENCY = 190.0

    def log_event(self, event_type: str, message: str, data: Dict[str, Any] = None):
        """记录事件"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "message": message,
            "data": data or {}
        }
        self.incident_log.append(entry)

    async def monitor_and_respond(self):
        """主监控循环"""
        print("\n" + "="*70)
        print("🔬 灾难恢复演示：高温反应器热失控场景")
        print("="*70)

        print("\n📊 初始状态:")
        state = self.reactor.read_state()
        print(f"   温度: {state.telemetry['temperature']}°C")
        print(f"   设定点: {state.telemetry['setpoint']}°C")
        print(f"   冷却流量: {state.telemetry['cooling_flow']*100}%")

        print("\n⏱️  开始模拟...")
        print("-"*70)

        # Phase 1: 正常运行
        print("\n[T+0s - T+30s] 🟢 正常运行阶段")
        for i in range(30):
            self.reactor.tick(1.0)
            if i % 10 == 0:
                state = self.reactor.read_state()
                print(f"   T+{i}s: 温度={state.telemetry['temperature']:.1f}°C ✓")
            await asyncio.sleep(0.02)

        # Phase 2: 注入故障
        print("\n[T+30s] 🔴 注入故障: 冷却系统失效")
        self.reactor.inject_fault()
        self.log_event("FAULT_INJECTED", "Cooling system fault simulated")

        # Phase 3: 故障发展
        print("\n[T+30s - T+90s] 🟡 故障发展阶段")
        for i in range(60):
            self.reactor.tick(1.0)
            t = 30 + i
            state = self.reactor.read_state()
            temp = state.telemetry['temperature']

            if i % 15 == 0:
                print(f"   T+{t}s: 温度={temp:.1f}°C (上升中...)")

            # 检测异常
            if temp > self.TEMP_WARNING and i == 30:
                print(f"\n   ⚠️  [SENSE] 检测到温度异常: {temp:.1f}°C > {self.TEMP_WARNING}°C")
                await self.handle_warning(state)

            await asyncio.sleep(0.02)

        # Phase 4: 临界状态 - 触发完整恢复流程
        print("\n[T+90s - T+120s] 🔴 临界状态 - 启动恢复流程")
        for i in range(30):
            self.reactor.tick(1.0)
            t = 90 + i
            state = self.reactor.read_state()
            temp = state.telemetry['temperature']

            if temp > self.TEMP_CRITICAL and not self.reactor.emergency_stop:
                print(f"\n   🚨 [CRITICAL] 温度达到临界值: {temp:.1f}°C")
                await self.execute_full_recovery(state)
                break

            await asyncio.sleep(0.02)

        # Phase 5: 恢复后监控
        print("\n[T+120s+] 🟢 恢复后监控")
        for i in range(30):
            self.reactor.tick(1.0)
            if i % 10 == 0:
                state = self.reactor.read_state()
                print(f"   温度={state.telemetry['temperature']:.1f}°C (降温中...)")
            await asyncio.sleep(0.02)

        # 最终状态
        print("\n" + "="*70)
        print("📋 最终状态报告")
        print("="*70)
        final_state = self.reactor.read_state()
        print(f"   最终温度: {final_state.telemetry['temperature']:.1f}°C")
        print(f"   紧急停机: {'是' if final_state.telemetry['emergency_stop'] else '否'}")
        print(f"   设备状态: {final_state.status}")
        print(f"   事件记录: {len(self.incident_log)} 条")

        # 打印事件摘要
        print("\n📜 事件日志摘要:")
        for event in self.incident_log[-5:]:
            print(f"   [{event['type']}] {event['message']}")

    async def handle_warning(self, state: DeviceState):
        """处理警告级别异常"""
        print("\n   --- 警告处理流程 ---")

        # Step 1: CLASSIFY - 分析故障签名
        print("   [CLASSIFY] 分析遥测数据签名...")
        history = self.reactor.get_history()
        signature = analyze_signature(history)
        print(f"   → 检测到模式: {signature.mode} (置信度: {signature.confidence:.2f})")
        self.log_event("SIGNATURE_DETECTED", f"Pattern: {signature.mode}",
                      {"confidence": signature.confidence})

        # Step 2: 调用诊断脚本
        print("   [DIAGNOSE] 运行诊断脚本...")
        diagnosis = await self.external.run_diagnostics(self.reactor.name)
        print(f"   → 诊断结果: {diagnosis['diagnosis']} (置信度: {diagnosis['confidence']:.2f})")
        self.log_event("DIAGNOSIS_COMPLETE", diagnosis['diagnosis'], diagnosis)

        # Step 3: 尝试软恢复 - 降低加热功率
        print("   [RECOVER] 尝试软恢复: 降低加热功率...")
        self.reactor.heater_power = 0.3
        self.log_event("SOFT_RECOVERY", "Reduced heater power to 30%")

    async def execute_full_recovery(self, state: DeviceState):
        """执行完整的灾难恢复流程"""
        print("\n" + "="*50)
        print("🚨 启动完整灾难恢复流程")
        print("="*50)

        # ====== STEP 1: 紧急停机 (CLI) ======
        print("\n[STEP 1/6] 🛑 紧急停机")
        print("   执行: labctl emergency-stop --device reactor_01")

        # 模拟CLI调用
        result = await self.external.emergency_shutdown(self.reactor.name)
        self.reactor.emergency_stop = True
        self.reactor.heater_power = 0

        print(f"   ✓ 停机完成: {result}")
        self.log_event("EMERGENCY_STOP", "Emergency shutdown executed", result)

        # ====== STEP 2: 创建硬件错误 ======
        print("\n[STEP 2/6] 📝 记录硬件错误")
        error = HardwareError(
            device=self.reactor.name,
            type="overshoot",
            severity="high",
            message=f"Temperature {state.telemetry['temperature']:.1f}°C exceeded critical threshold {self.TEMP_CRITICAL}°C",
            when=datetime.now().isoformat(),
            action="heat_control",
            context={
                "cooling_fault": True,
                "cooling_flow": state.telemetry['cooling_flow'],
                "signature": "drift"
            }
        )
        print(f"   ✓ 错误记录: {error}")

        # ====== STEP 3: 策略决策 ======
        print("\n[STEP 3/6] 🧠 执行恢复策略决策")
        history = self.reactor.get_history()

        decision = decide_recovery(
            state=state,
            error=error,
            history=history,
            retry_counts=self.retry_counts,
            last_action=Action(name="heat_control", effect="write", params={"temperature": 150}),
            stage="reaction",
            config=self.recovery_config
        )

        print(f"   决策类型: {decision.kind}")
        print(f"   决策理由: {decision.rationale}")
        print(f"   恢复动作: {len(decision.actions)} 个")
        for action in decision.actions:
            print(f"      - {action.name}: {action.params}")
        self.log_event("RECOVERY_DECISION", decision.rationale,
                      {"kind": decision.kind, "actions": len(decision.actions)})

        # ====== STEP 4: 数据备份 (Script) ======
        print("\n[STEP 4/6] 💾 备份实验数据")
        backup_result = await self.external.backup_experiment_data("EXP-2024-001")
        print(f"   ✓ 备份完成: {backup_result['backup_path']}")
        print(f"   ✓ 备份大小: {backup_result['size_mb']} MB")
        self.log_event("DATA_BACKUP", "Experiment data backed up", backup_result)

        # ====== STEP 5: 上报LIMS (API) ======
        print("\n[STEP 5/6] 📡 上报LIMS系统")
        incident = {
            "device": self.reactor.name,
            "error_type": error.type,
            "severity": error.severity,
            "temperature_max": state.telemetry['temperature'],
            "diagnosis": "cooling_pump_failure",
            "decision": decision.kind,
            "timestamp": datetime.now().isoformat()
        }

        lims_result = await self.external.report_to_lims(incident)
        print(f"   ✓ 事件ID: {lims_result['incident_id']}")
        print(f"   ✓ 分配给: {lims_result['assigned_to']}")
        self.log_event("LIMS_REPORTED", f"Incident {lims_result['incident_id']}", lims_result)

        # ====== STEP 6: 发送告警 (API) ======
        print("\n[STEP 6/6] 📱 发送告警通知")
        alert_message = (
            f"🚨 紧急: {self.reactor.name} 热失控\n"
            f"温度: {state.telemetry['temperature']:.1f}°C\n"
            f"状态: 已执行紧急停机\n"
            f"事件ID: {lims_result['incident_id']}"
        )

        alert_result = await self.external.send_alert(alert_message, "critical")
        print(f"   ✓ 已通知: {', '.join(alert_result['channels'])}")
        self.log_event("ALERT_SENT", "Critical alert sent", alert_result)

        print("\n" + "="*50)
        print("✅ 灾难恢复流程完成")
        print("="*50)


# ============================================================================
# 真实外部系统集成示例（可选）
# ============================================================================

async def demo_real_external_calls():
    """演示真实的外部系统调用（不执行，仅展示代码）"""
    print("\n" + "="*70)
    print("📖 真实外部系统集成代码示例")
    print("="*70)

    print("""
# 1. CLI工具调用 - 真实代码
from exp_agent.external.cli import CLIExecutor, CLIAction

cli = CLIExecutor(allowed_commands=["labctl", "docker"])
result = await cli.execute(CLIAction(
    name="emergency_stop",
    command="labctl",
    args=["emergency-stop", "--device", "reactor_01", "--reason", "thermal_runaway"],
    timeout_seconds=10,
    retries=2
))

# 2. 本地脚本执行 - 真实代码
from exp_agent.external.script import ScriptExecutor, ScriptAction, ScriptType

script = ScriptExecutor(python_venv="/lab/.venv")
result = await script.execute(ScriptAction(
    name="run_diagnostics",
    script_path="/lab/scripts/diagnose_cooling.py",
    script_type=ScriptType.PYTHON,
    args=["--device", "reactor_01", "--output", "json"],
    timeout_seconds=30
))

# 3. 外部API调用 - 真实代码
from exp_agent.external.api import APIExecutor, APIAction, HTTPMethod

api = APIExecutor(
    base_url="https://lims.lab.com/api/v1",
    default_headers={"Authorization": "Bearer <token>"}
)
result = await api.execute(APIAction(
    name="report_incident",
    url="/incidents",
    method=HTTPMethod.POST,
    json_body={
        "device": "reactor_01",
        "type": "thermal_runaway",
        "severity": "critical",
        "auto_reported": True
    },
    retries=3,
    retry_on_status=[500, 502, 503]
))

# 4. 集成为设备使用
from exp_agent.external.integration import create_hybrid_device, ExternalAction

lab_system = create_hybrid_device(
    name="lab_control",
    cli_config={"allowed_commands": ["labctl", "spectra-cli"]},
    script_config={"python_venv": "/lab/.venv", "allowed_paths": ["/lab/scripts"]},
    api_config={"base_url": "https://lims.lab.com/api/v1"}
)

# 通过统一接口执行
action = ExternalAction(
    name="full_shutdown",
    effect="write",
    executor_type="cli",
    external_config={"command": "labctl", "args": ["shutdown", "--all"]}
)
lab_system.execute(action.to_core_action())
""")


# ============================================================================
# Main
# ============================================================================

async def main():
    """主函数"""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " 🔬 EXP-AGENT 灾难恢复演示 ".center(68) + "║")
    print("║" + " Disaster Recovery Demo with External System Integration ".center(68) + "║")
    print("╚" + "═"*68 + "╝")

    if HAS_RUST:
        print("\n✅ Rust telemetry module loaded")
    else:
        print("\n⚠️  Using Python fallback (run 'maturin develop' to enable Rust)")

    # 创建模拟反应器
    reactor = SimulatedReactor()

    # 创建恢复协调器
    coordinator = DisasterRecoveryCoordinator(reactor)

    # 运行演示
    await coordinator.monitor_and_respond()

    # 展示真实代码示例
    await demo_real_external_calls()

    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
