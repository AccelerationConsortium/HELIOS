from __future__ import annotations


async def test_recovery_agent_creates_episode_for_first_retry():
    from app.agents.recovery_agent import RecoveryAgent, RecoveryInput

    result = await RecoveryAgent().run(
        RecoveryInput(
            error_type="communication_error",
            error_message="serial read timed out",
            device_name="pump",
            retry_count=0,
            telemetry={"run_id": "run-1", "step_key": "pump.flush"},
        )
    )

    assert result.success is True
    assert result.output is not None
    assert result.output.decision == "retry"
    assert result.output.phase == "plan_attempt"
    assert result.output.episode.run_id == "run-1"
    assert result.output.episode.step_key == "pump.flush"
    assert result.output.episode.observations[-1]["error_type"] == "communication_error"
    assert result.output.episode.attempts[-1].action == "retry_original"
    assert result.output.next_action is not None


async def test_recovery_agent_revises_episode_after_failed_attempt():
    from app.agents.recovery_agent import RecoveryAgent, RecoveryInput

    agent = RecoveryAgent()
    first = await agent.run(
        RecoveryInput(
            error_type="communication_error",
            error_message="serial read timed out",
            device_name="pump",
            retry_count=0,
        )
    )
    assert first.output is not None

    second = await agent.run(
        RecoveryInput(
            error_type="communication_error",
            error_message="serial read timed out again",
            device_name="pump",
            retry_count=1,
            episode=first.output.episode,
            last_attempt_result={
                "result": "failed",
                "error_type": "communication_error",
            },
        )
    )

    assert second.success is True
    assert second.output is not None
    episode = second.output.episode
    assert len(episode.observations) == 2
    assert episode.attempts[0].result == "failed"
    assert episode.attempts[-1].action == "wait_and_retry"
    assert "transient communication failure" in episode.rejected_hypotheses
