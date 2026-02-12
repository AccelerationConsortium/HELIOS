from exp_agent.recovery.recovery_agent import RecoveryAgent
from exp_agent.llm.stub_advisor import EchoBaselineAdvisor
from exp_agent.core.types import DeviceState, HardwareError, Action


def test_recovery_agent_accepts_llm_advisor_and_sets_last_proposal():
    agent = RecoveryAgent(llm_advisor=EchoBaselineAdvisor())

    state = DeviceState(name="heater_1", status="running", telemetry={"temperature": 120})
    err = HardwareError(device="heater_1", type="timeout", severity="medium", message="timeout")
    last_action = Action(name="set_temperature", effect="write", params={"temperature": 120})

    decision = agent.decide(state=state, error=err, history=[state], last_action=last_action, stage="heating")

    assert decision.kind in {"retry", "degrade", "abort", "skip"}
    assert agent.last_llm_proposal is not None
    assert agent.last_llm_proposal.source == "llm"
    assert "LLM stub" in agent.last_llm_proposal.rationale
