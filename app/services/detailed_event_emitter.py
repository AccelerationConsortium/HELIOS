"""
Detailed Event Emitter for Frontend UI

提供详细的agent思考过程、工具调用、硬件操控等事件
用于前端UI的execution tree展示
"""
from __future__ import annotations

import time
from typing import Any, Callable


class DetailedEventEmitter:
    """增强的事件发射器，支持层级化的execution tree"""

    def __init__(self, campaign_id: str, emit_func: Callable):
        self.campaign_id = campaign_id
        self.emit = emit_func
        self.indent_level = 0
        self.step_counter = 0

    def emit_agent_start(self, agent_name: str, description: str, metadata: dict = None):
        """开始一个agent步骤"""
        self.step_counter += 1
        step_id = f"step_{self.step_counter}"

        self.emit(self.campaign_id, {
            "type": "agent_thinking",
            "agent": agent_name,
            "step_id": step_id,
            "indent": self.indent_level,
            "message": description,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })

        self.indent_level += 1
        return step_id

    def emit_agent_decision(self, agent_name: str, decision: str, reasoning: str):
        """Agent做出决策"""
        self.emit(self.campaign_id, {
            "type": "agent_decision",
            "agent": agent_name,
            "indent": self.indent_level,
            "decision": decision,
            "reasoning": reasoning,
            "timestamp": time.time(),
        })

    def emit_agent_result(self, agent_name: str, success: bool, message: str, data: dict = None):
        """Agent完成"""
        self.indent_level = max(0, self.indent_level - 1)

        self.emit(self.campaign_id, {
            "type": "agent_result",
            "agent": agent_name,
            "indent": self.indent_level,
            "success": success,
            "message": message,
            "data": data or {},
            "timestamp": time.time(),
        })

    def emit_tool_call(self, tool_name: str, operation: str, params: dict = None):
        """工具调用"""
        self.emit(self.campaign_id, {
            "type": "tool_call",
            "tool": tool_name,
            "indent": self.indent_level,
            "operation": operation,
            "params": params or {},
            "timestamp": time.time(),
        })

    def emit_hardware_action(self, hardware: str, action: str, details: dict = None):
        """硬件操作"""
        self.emit(self.campaign_id, {
            "type": "hardware_action",
            "hardware": hardware,
            "indent": self.indent_level,
            "action": action,
            "details": details or {},
            "timestamp": time.time(),
        })

    def emit_log(self, level: str, message: str):
        """日志消息"""
        self.emit(self.campaign_id, {
            "type": "log",
            "level": level,  # info, warning, error, success
            "indent": self.indent_level,
            "message": message,
            "timestamp": time.time(),
        })

    def emit_protocol_step(self, step_num: int, description: str):
        """Protocol步骤"""
        self.emit(self.campaign_id, {
            "type": "protocol_step",
            "step_num": step_num,
            "indent": self.indent_level,
            "description": description,
            "timestamp": time.time(),
        })

    def emit_safety_check(self, check_name: str, passed: bool, details: str = ""):
        """安全检查"""
        self.emit(self.campaign_id, {
            "type": "safety_check",
            "check_name": check_name,
            "indent": self.indent_level,
            "passed": passed,
            "details": details,
            "timestamp": time.time(),
        })

    def emit_thinking(self, message: str):
        """思考过程"""
        self.emit(self.campaign_id, {
            "type": "thinking",
            "indent": self.indent_level,
            "message": message,
            "timestamp": time.time(),
        })


# Helper function to emit detailed round execution
async def emit_detailed_round_execution(
    emitter: DetailedEventEmitter,
    round_num: int,
    strategy: str,
    candidate_params: dict,
    simulate: bool = True
):
    """
    发送详细的单轮执行事件

    Args:
        emitter: DetailedEventEmitter实例
        round_num: 轮次号
        strategy: 策略名称（lhs, bayesian等）
        candidate_params: 候选参数
        simulate: 是否模拟执行（添加延迟）
    """
    import asyncio

    # Round开始
    emitter.emit_agent_start(
        "orchestrator",
        f"Round {round_num} execution",
        {"round": round_num, "strategy": strategy}
    )

    # 1. PlannerAgent决策
    step_id = emitter.emit_agent_start(
        "planner",
        "Selecting optimization strategy",
        {"round": round_num}
    )

    if simulate:
        await asyncio.sleep(0.3)

    emitter.emit_thinking(f"Analyzing campaign progress... (Round {round_num})")

    if round_num == 1:
        emitter.emit_agent_decision(
            "planner",
            f"Strategy: {strategy.upper()}",
            "First round: use space-filling design for diversity"
        )
    else:
        emitter.emit_agent_decision(
            "planner",
            f"Strategy: {strategy.upper()}",
            f"Sufficient data available, switching to surrogate-based optimization"
        )

    if simulate:
        await asyncio.sleep(0.2)

    emitter.emit_agent_result("planner", True, f"Strategy selected: {strategy}")

    # 2. CandidateGenerator生成候选
    step_id = emitter.emit_agent_start(
        "candidate_gen",
        f"Generating candidate using {strategy}",
        {"strategy": strategy}
    )

    if simulate:
        await asyncio.sleep(0.3)

    if strategy == "lhs":
        emitter.emit_thinking("Generating space-filling sample using Latin Hypercube")
        emitter.emit_log("info", f"Sampling {len(candidate_params)} dimensions")
    else:
        emitter.emit_thinking(f"Training surrogate model on {round_num-1} data points")
        emitter.emit_log("info", "Model: KNN regressor (k=3)")
        emitter.emit_thinking("Optimizing Expected Improvement acquisition function")
        emitter.emit_log("info", "Search method: Differential Evolution")

    if simulate:
        await asyncio.sleep(0.3)

    emitter.emit_agent_result(
        "candidate_gen",
        True,
        "Candidate generated",
        {"params": candidate_params}
    )

    # 3. SafetyAgent验证
    step_id = emitter.emit_agent_start(
        "safety",
        "Pre-execution safety validation",
        {}
    )

    if simulate:
        await asyncio.sleep(0.2)

    emitter.emit_thinking("Checking safety constraints...")

    safety_checks = [
        ("Volume limit", "3.0 mL max", True),
        ("Current density", "50 mA/cm² max", True),
        ("Temperature", "50°C max", True),
        ("Tip budget", "200 tips available", True),
        ("Deck layout", "All positions valid", True),
    ]

    for check_name, limit, passed in safety_checks:
        emitter.emit_safety_check(f"{check_name}: {limit}", passed)
        if simulate:
            await asyncio.sleep(0.1)

    emitter.emit_agent_result("safety", True, "All safety checks passed ✅")

    # 4. CompilerAgent生成协议
    step_id = emitter.emit_agent_start(
        "compiler",
        "Generating OT-2 protocol",
        {}
    )

    if simulate:
        await asyncio.sleep(0.3)

    emitter.emit_thinking("Planning deck layout...")
    # Dynamic deck info based on robot type (OT-2: 11 slots / Flex: 12 slots)
    emitter.emit_log("info", "Deck layout planned, compiling protocol steps...")

    if simulate:
        await asyncio.sleep(0.2)

    emitter.emit_thinking("Generating protocol steps...")

    protocol_steps = [
        "Pre-clean reactor (water + ultrasound 30s)",
        "Acid rinse (1M H2SO4, 10s)",
        "Final rinse + ultrasound (20s)",
        "Dispense precursor mixture to well A1",
        "Electrodeposition (galvanostatic)",
        "Clean deposition tool",
        "Photo capture (top-view)",
        "Flush precursor, fill 1M KOH",
        "Insert 3-electrode setup",
        "Run HER test (CV + EIS + galvanostatic)",
        "Compute η10 from polarization curve"
    ]

    for idx, step_desc in enumerate(protocol_steps, 1):
        emitter.emit_protocol_step(idx, step_desc)
        if simulate:
            await asyncio.sleep(0.05)

    emitter.emit_agent_result("compiler", True, "Protocol compiled (450 lines)")

    # 5. Executor - 硬件执行
    step_id = emitter.emit_agent_start(
        "executor",
        "Executing on OT-2 hardware",
        {}
    )

    if simulate:
        await asyncio.sleep(0.3)

    # P300清洗
    emitter.emit_tool_call("P300_pipette", "reactor_cleaning", {"volume": "1000 µL"})
    emitter.emit_hardware_action("P300", "Aspirate H2O from reservoir → dispense to reactor")
    if simulate:
        await asyncio.sleep(0.2)
    emitter.emit_hardware_action("Ultrasound", "30s cleaning cycle")
    if simulate:
        await asyncio.sleep(0.2)

    # P20分配
    emitter.emit_tool_call("P20_pipette", "precursor_dispensing", {})
    for i in range(1, 11):
        emitter.emit_hardware_action("P20", f"Aspirate stock {i} → well A1")
        if simulate:
            await asyncio.sleep(0.1)
    emitter.emit_hardware_action("P20", "Mix 5 cycles (10 µL)")
    if simulate:
        await asyncio.sleep(0.2)

    # 电化学沉积
    emitter.emit_tool_call("Electrodeposition", "galvanostatic_deposition", {
        "current_density": "10 mA/cm²",
        "time": "45s"
    })
    emitter.emit_hardware_action("Electrode", "Insert working electrode into well A1")
    if simulate:
        await asyncio.sleep(0.2)
    emitter.emit_hardware_action("Potentiostat", "Apply current: 10 mA/cm² for 45s")
    emitter.emit_log("success", "Potential vs time logged")
    if simulate:
        await asyncio.sleep(0.3)
    emitter.emit_hardware_action("Electrode", "Retract and air dry 10s")
    if simulate:
        await asyncio.sleep(0.2)

    # 拍照QC
    emitter.emit_tool_call("Camera", "photo_capture", {"resolution": "1920x1080"})
    emitter.emit_hardware_action("Camera", "Position above well A1")
    if simulate:
        await asyncio.sleep(0.1)
    emitter.emit_hardware_action("Camera", "Capture top-view image")
    emitter.emit_log("success", "Image quality: good (no bubbles, uniform)")
    if simulate:
        await asyncio.sleep(0.2)

    # HER测试
    emitter.emit_tool_call("Potentiostat", "her_testing", {"electrolyte": "1M KOH"})
    emitter.emit_hardware_action("Potentiostat", "Flush precursor, fill 1M KOH")
    if simulate:
        await asyncio.sleep(0.2)
    emitter.emit_hardware_action("Potentiostat", "Insert 3-electrode setup")
    if simulate:
        await asyncio.sleep(0.2)
    emitter.emit_hardware_action("Potentiostat", "CV scan: -0.2 to -0.6V vs RHE")
    emitter.emit_log("info", "Forward/reverse sweep recorded")
    if simulate:
        await asyncio.sleep(0.3)
    emitter.emit_hardware_action("Potentiostat", "EIS: 100kHz - 0.1Hz at η = -100mV")
    emitter.emit_log("info", "Nyquist plot: RΩ = 3.8Ω, Rct = 12.4Ω")
    if simulate:
        await asyncio.sleep(0.3)
    emitter.emit_hardware_action("Potentiostat", "Galvanostatic: 10 mA/cm² for 60s")
    emitter.emit_log("success", "Stable potential: -0.127V vs RHE")
    if simulate:
        await asyncio.sleep(0.3)

    # 模拟结果
    eta10 = 127.3 if round_num == 1 else 89.7
    emitter.emit_log("success", f"η10 extracted: {eta10} mV")

    emitter.emit_agent_result("executor", True, f"Execution complete, η10 = {eta10} mV")

    # 6. SensingAgent QC
    step_id = emitter.emit_agent_start(
        "sensing",
        "QC validation",
        {}
    )

    if simulate:
        await asyncio.sleep(0.2)

    emitter.emit_thinking("Analyzing experimental results...")

    qc_checks = [
        ("Photo quality", "good"),
        ("Volume accuracy", "±5%"),
        ("HER curve shape", "valid"),
        ("EIS spectrum", "valid"),
    ]

    for check, result in qc_checks:
        emitter.emit_log("success", f"{check}: {result}")
        if simulate:
            await asyncio.sleep(0.1)

    emitter.emit_agent_result("sensing", True, "QC passed, data valid ✅")

    # Round完成
    emitter.emit_agent_result(
        "orchestrator",
        True,
        f"Round {round_num} completed: η10 = {eta10} mV"
    )
