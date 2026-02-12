"""Recovery Agent Demo - Example usage scenarios.

This script demonstrates RecoveryAgent capabilities:
1. Basic error recovery with retries
2. Chemical safety event handling
3. Fault signature analysis with history
4. Integration patterns for orchestrator
"""
import asyncio
from app.agents import RecoveryAgent, RecoveryInput


async def demo_basic_recovery():
    """Demo 1: Basic timeout error recovery."""
    print("\n" + "="*60)
    print("Demo 1: Basic Timeout Recovery")
    print("="*60)

    agent = RecoveryAgent()

    # Simulate a timeout error
    input_data = RecoveryInput(
        error_type="timeout",
        error_message="Connection timeout after 30s",
        device_name="opentrons_ot2",
        device_status="error",
        error_severity="low",
        telemetry={"last_response_time": 30.5},
        retry_count=0,
    )

    result = await agent.run(input_data)

    if result.success:
        print(f"✅ Decision: {result.output.decision}")
        print(f"📝 Rationale: {result.output.rationale}")
        if result.output.retry_delay_seconds > 0:
            print(f"⏱️  Retry delay: {result.output.retry_delay_seconds}s")
        print(f"🔢 Max retries: {result.output.max_retries}")
    else:
        print(f"❌ Recovery failed: {result.errors}")


async def demo_chemical_safety():
    """Demo 2: Chemical safety event escalation."""
    print("\n" + "="*60)
    print("Demo 2: Chemical Safety Event")
    print("="*60)

    agent = RecoveryAgent()

    # Simulate a spill detection
    input_data = RecoveryInput(
        error_type="spill_detected",
        error_message="Liquid spill detected in workspace",
        device_name="opentrons_ot2",
        device_status="error",
        error_severity="high",
        telemetry={
            "spill_detected": True,
            "volume_ml": 5.0,
            "temperature": 25.0,
        },
        retry_count=0,
    )

    result = await agent.run(input_data)

    if result.success:
        print(f"🚨 Chemical Safety Event: {result.output.chemical_safety_event}")
        print(f"⛔ Decision: {result.output.decision}")
        print(f"📝 Rationale: {result.output.rationale}")
        print(f"🛡️  SafetyAgent veto power: ACTIVE")
        if result.output.actions:
            print(f"🔧 Actions: {len(result.output.actions)} emergency actions")
            for action in result.output.actions:
                print(f"   • {action['name']}")
    else:
        print(f"❌ Recovery failed: {result.errors}")


async def demo_sensor_drift():
    """Demo 3: Sensor drift with telemetry history."""
    print("\n" + "="*60)
    print("Demo 3: Sensor Drift Analysis")
    print("="*60)

    agent = RecoveryAgent()

    # Simulate temperature sensor drift with history
    input_data = RecoveryInput(
        error_type="sensor_drift",
        error_message="Temperature sensor reading drift detected",
        device_name="plc_controller",
        device_status="error",
        error_severity="medium",
        telemetry={"temperature": 45.0, "drift_rate": 2.5},
        history=[
            {
                "device_name": "plc_controller",
                "status": "idle",
                "telemetry": {"temperature": 25.0, "drift_rate": 0.0}
            },
            {
                "device_name": "plc_controller",
                "status": "running",
                "telemetry": {"temperature": 30.0, "drift_rate": 0.5}
            },
            {
                "device_name": "plc_controller",
                "status": "running",
                "telemetry": {"temperature": 35.0, "drift_rate": 1.0}
            },
            {
                "device_name": "plc_controller",
                "status": "error",
                "telemetry": {"temperature": 40.0, "drift_rate": 1.5}
            },
            {
                "device_name": "plc_controller",
                "status": "error",
                "telemetry": {"temperature": 45.0, "drift_rate": 2.5}
            },
        ],
        retry_count=1,
    )

    result = await agent.run(input_data)

    if result.success:
        print(f"✅ Decision: {result.output.decision}")
        print(f"📝 Rationale: {result.output.rationale}")
        print(f"📊 History analyzed: {len(input_data.history)} data points")
        print(f"🌡️  Temperature drift: 25°C → 45°C (+20°C)")
        print(f"📈 Drift rate: 2.5°C/step")
    else:
        print(f"❌ Recovery failed: {result.errors}")


async def demo_orchestrator_pattern():
    """Demo 4: Orchestrator integration pattern."""
    print("\n" + "="*60)
    print("Demo 4: Orchestrator Integration Pattern")
    print("="*60)

    agent = RecoveryAgent()

    async def simulate_execution_with_recovery(attempt: int = 0):
        """Simulate execution with potential failure."""
        print(f"\n🔄 Execution attempt {attempt + 1}")

        # Simulate random failure
        import random
        if random.random() < 0.7 and attempt < 2:  # 70% failure rate for first 2 attempts
            error_type = random.choice(["timeout", "sensor_fail", "actuator_jam"])
            print(f"   ❌ Execution failed: {error_type}")

            # Get recovery decision
            recovery_input = RecoveryInput(
                error_type=error_type,
                error_message=f"{error_type} occurred during execution",
                device_name="opentrons_ot2",
                device_status="error",
                error_severity="medium",
                retry_count=attempt,
            )

            recovery_result = await agent.run(recovery_input)

            if recovery_result.success:
                decision = recovery_result.output.decision
                print(f"   🤔 Recovery decision: {decision}")
                print(f"   💡 Rationale: {recovery_result.output.rationale[:60]}...")

                if decision == "retry":
                    delay = recovery_result.output.retry_delay_seconds
                    print(f"   ⏳ Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                    return await simulate_execution_with_recovery(attempt + 1)

                elif decision == "abort":
                    print(f"   ⛔ Aborting execution")
                    return None

                elif decision == "degrade":
                    print(f"   ⚠️  Continuing in degraded mode")
                    return {"kpi": 85.0, "degraded": True}

                else:  # skip
                    print(f"   ⏭️  Skipping this execution")
                    return None
            else:
                print(f"   ❌ Recovery agent failed: {recovery_result.errors}")
                return None

        else:
            print(f"   ✅ Execution successful")
            return {"kpi": 95.0, "degraded": False}

    # Simulate orchestrator execution loop
    result = await simulate_execution_with_recovery()

    if result:
        print(f"\n🎉 Final result: KPI={result['kpi']}, degraded={result['degraded']}")
    else:
        print(f"\n❌ Execution failed after recovery attempts")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("🤖 Recovery Agent Demo")
    print("="*60)

    agent = RecoveryAgent()
    if agent._available:
        print("✅ Full recovery-agent capabilities active")
    else:
        print("⚠️  Using fallback recovery logic")

    await demo_basic_recovery()
    await demo_chemical_safety()
    await demo_sensor_drift()
    await demo_orchestrator_pattern()

    print("\n" + "="*60)
    print("✅ All demos completed")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
