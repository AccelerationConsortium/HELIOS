"""Strategy Router — RL / rule-based routing with A/B test and online learning.

The router sits between the orchestrator and the strategy selector,
deciding *which* selection method to use (rule-based expert vs. RL agent)
and managing per-campaign RL state (transitions, rewards, learning).

Modes:
- **rule_based**: Always use the hand-crafted expert system (default, safe)
- **rl**: Always use the RL agent (with safety fallback on low confidence)
- **ab_test**: Deterministically assign campaigns to rule-based or RL
  based on campaign_id hash, then log results for offline comparison

Safety guarantees:
- If RL confidence < threshold → fall back to rule-based
- If RL raises any exception → fall back to rule-based
- A/B test uses deterministic hashing (reproducible assignment)
- All RL errors are silently caught (never crash the campaign)
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["StrategyRouter", "RouterConfig"]


# ---------------------------------------------------------------------------
# Router Config
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    """Configuration for the strategy router."""

    # Mode: "rule_based" | "rl" | "ab_test"
    mode: str = "rule_based"

    # RL backend: "dqn" | "ppo" | "q_learning"
    rl_backend: str = "dqn"

    # A/B test settings
    ab_test_rl_fraction: float = 0.2  # Fraction of campaigns routed to RL

    # Safety
    fallback_on_error: bool = True  # Fall back to rule-based if RL fails
    confidence_threshold: float = 0.3  # Minimum RL confidence to use RL action

    # Online learning
    online_learning: bool = True  # Train RL agent on each round

    # Persistence
    save_frequency: int = 5  # Save model every N campaigns
    model_path: str = "models/dqn_selector.pth"

    # Exploration
    explore: bool = True  # Enable ε-greedy exploration


# ---------------------------------------------------------------------------
# Per-campaign RL state
# ---------------------------------------------------------------------------

@dataclass
class _CampaignRLState:
    """Tracks per-campaign RL state for online learning."""

    treatment: str = "rule_based"  # "rule_based" | "rl_dqn" | "rl_ppo" | "rl_q_learning"
    prev_state: Any = None  # RLState from previous round
    prev_action: int | None = None  # Action taken in previous round
    action_history: list[int] = field(default_factory=list)
    reward_trace: list[float] = field(default_factory=list)
    prev_best_kpi: float | None = None


# ---------------------------------------------------------------------------
# Strategy Router
# ---------------------------------------------------------------------------

class StrategyRouter:
    """Routes strategy selection between RL and rule-based systems.

    Usage in orchestrator:
        router = StrategyRouter(RouterConfig(mode="ab_test"))

        # At strategy selection time:
        decision = router.select_strategy(snapshot, campaign_id)

        # After each round completes:
        router.on_round_complete(campaign_id, snapshot, diagnostics, ...)

        # After campaign ends:
        router.on_campaign_complete(campaign_id)
    """

    def __init__(self, config: RouterConfig | None = None):
        if config is None:
            config = RouterConfig()
        self.config = config

        # Lazy-initialized RL selectors
        self._rl_selector: Any = None
        self._initialized_backend: str | None = None

        # Per-campaign state
        self._campaign_states: dict[str, _CampaignRLState] = {}

        # A/B test logger
        self._ab_logger: Any = None

        # Campaign counter (for periodic save)
        self._campaigns_completed = 0

        # Adaptive reward components (Phase 2)
        self._reward_weight_learner: Any = None
        self._reward_model: Any = None
        self._reward_model_trainer: Any = None
        self._reward_config: Any = None  # Current RewardConfig with learned weights

        # Initialize adaptive reward components
        try:
            from app.services.reward_weight_learner import RewardWeightLearner
            self._reward_weight_learner = RewardWeightLearner()
            self._reward_config = self._reward_weight_learner.get_reward_config()
            logger.info("Reward weight learner initialized")
        except Exception:
            logger.debug("Reward weight learner not available", exc_info=True)

        try:
            from app.services.reward_model import RewardModel, RewardModelTrainer
            self._reward_model = RewardModel()
            self._reward_model_trainer = RewardModelTrainer(self._reward_model)
            logger.info("Neural reward model initialized (λ=%.3f)", self._reward_model.lambda_blend)
        except Exception:
            logger.debug("Neural reward model not available", exc_info=True)

        logger.info(
            "StrategyRouter initialized: mode=%s, rl_backend=%s, "
            "ab_test_rl_fraction=%.2f, confidence_threshold=%.2f",
            config.mode, config.rl_backend,
            config.ab_test_rl_fraction, config.confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_strategy(
        self,
        snapshot: Any,
        campaign_id: str,
    ) -> Any:
        """Select strategy, routing through RL or rule-based as configured.

        Args:
            snapshot: CampaignSnapshot with current campaign state
            campaign_id: Unique campaign identifier

        Returns:
            StrategyDecision from either RL or rule-based selector
        """
        from app.services.strategy_selector import (
            CampaignSnapshot,
            StrategyDecision,
            compute_diagnostics,
            select_strategy as rule_based_select,
        )

        # Ensure per-campaign state exists
        if campaign_id not in self._campaign_states:
            self._campaign_states[campaign_id] = _CampaignRLState()

        state = self._campaign_states[campaign_id]

        # --- Determine treatment ---
        use_rl = self._should_use_rl(campaign_id)

        if not use_rl:
            # Rule-based path
            state.treatment = "rule_based"
            return rule_based_select(snapshot)

        # --- RL path ---
        state.treatment = f"rl_{self.config.rl_backend}"

        try:
            rl_selector = self._get_rl_selector()
            diagnostics = compute_diagnostics(snapshot)

            # Check confidence (safety gate)
            if hasattr(rl_selector, "get_confidence"):
                confidence = rl_selector.get_confidence(snapshot, diagnostics)
                if confidence < self.config.confidence_threshold:
                    logger.info(
                        "RL confidence %.3f < threshold %.3f for campaign=%s, "
                        "falling back to rule-based",
                        confidence, self.config.confidence_threshold, campaign_id,
                    )
                    state.treatment = "rule_based"
                    return rule_based_select(snapshot)

            # Select action via RL
            action_id, backend_name = rl_selector.select_action(
                snapshot, diagnostics, explore=self.config.explore,
            )

            # Track state for online learning
            from app.services.rl_strategy_selector import RLState
            current_rl_state = RLState.from_snapshot(snapshot, diagnostics)
            state.prev_state = current_rl_state
            state.prev_action = action_id
            state.action_history.append(action_id)

            # Build a StrategyDecision matching the rule-based API
            decision = self._build_rl_decision(
                action_id, backend_name, snapshot, diagnostics,
            )

            logger.info(
                "RL selected: campaign=%s action=%s backend=%s",
                campaign_id, action_id, backend_name,
            )
            return decision

        except Exception:
            if self.config.fallback_on_error:
                logger.debug(
                    "RL selector failed for campaign=%s, falling back to rule-based",
                    campaign_id, exc_info=True,
                )
                state.treatment = "rule_based"
                return rule_based_select(snapshot)
            raise

    def on_round_complete(
        self,
        campaign_id: str,
        snapshot: Any,
        diagnostics: Any,
        action: int | None,
        kpi_prev: float | None,
        kpi_curr: float | None,
        n_qc_failures: int = 0,
        is_terminal: bool = False,
        target_reached: bool = False,
    ) -> None:
        """Called after each round to update RL agent (online learning).

        This is a best-effort hook — errors are silently logged, never raised.
        """
        if not self.config.online_learning:
            return

        state = self._campaign_states.get(campaign_id)
        if state is None or not state.treatment.startswith("rl_"):
            return

        try:
            rl_selector = self._get_rl_selector()
            if rl_selector is None:
                return

            # Compute reward (with learned weights if available)
            from app.services.rl_reward import compute_reward, RewardConfig
            from app.services.rl_strategy_selector import RLState

            reward_components = compute_reward(
                snapshot=snapshot,
                diagnostics=diagnostics,
                kpi_prev=kpi_prev,
                kpi_curr=kpi_curr,
                n_qc_failures=n_qc_failures,
                action=action if action is not None else state.prev_action,
                prev_actions=state.action_history[:-1] if state.action_history else None,
                is_terminal=is_terminal,
                target_reached=target_reached,
                config=self._reward_config,  # Use learned weights if available
            )

            reward = reward_components.total

            # Compute current RL state for next_state
            current_rl_state = RLState.from_snapshot(snapshot, diagnostics)

            # Blend with neural reward model if available
            if self._reward_model is not None and state.prev_state is not None:
                try:
                    reward = self._reward_model.blend_reward(
                        handcrafted_reward=reward,
                        state=state.prev_state,
                        action=action if action is not None else state.prev_action,
                        next_state=None if is_terminal else current_rl_state,
                    )
                except Exception:
                    pass  # Silently fall back to handcrafted

            state.reward_trace.append(reward)
            state.prev_best_kpi = kpi_curr

            # Train on transition (if we have a previous state)
            if state.prev_state is not None and state.prev_action is not None:
                rl_selector.learn_from_experience(
                    state=state.prev_state,
                    action=state.prev_action,
                    reward=reward,
                    next_state=None if is_terminal else current_rl_state,
                    done=is_terminal,
                )

            # Update state for next round
            if not is_terminal:
                state.prev_state = current_rl_state

            logger.debug(
                "RL round update: campaign=%s reward=%.4f terminal=%s",
                campaign_id, reward, is_terminal,
            )

        except Exception:
            logger.debug(
                "RL on_round_complete failed for campaign=%s",
                campaign_id, exc_info=True,
            )

    def on_campaign_complete(self, campaign_id: str) -> None:
        """Called when campaign finishes. Logs A/B test, updates adaptive rewards, saves model.

        This is a best-effort hook — errors are silently logged, never raised.
        """
        state = self._campaign_states.get(campaign_id)
        if state is None:
            return

        try:
            # Log A/B test result
            if self.config.mode == "ab_test":
                self._log_ab_result(campaign_id, state)

            # --- Adaptive Reward: Update weight learner ---
            if self._reward_weight_learner is not None and state.treatment.startswith("rl_"):
                try:
                    from app.services.reward_weight_learner import CampaignOutcome
                    total_reward = sum(state.reward_trace) if state.reward_trace else 0.0
                    outcome = CampaignOutcome(
                        campaign_id=campaign_id,
                        n_rounds=len(state.action_history),
                        final_kpi=state.prev_best_kpi,
                        best_kpi=state.prev_best_kpi,
                        target_value=None,  # Not available here; could be passed in
                        direction="maximize",  # Default; could be from snapshot
                        target_reached=total_reward > 1.0,
                        converged=total_reward > 0,
                        reward_trace=state.reward_trace,
                    )
                    self._reward_config = self._reward_weight_learner.update_weights(outcome)
                    logger.debug("Updated reward weights for next campaign")
                except Exception:
                    logger.debug("Reward weight update failed", exc_info=True)

            # --- Adaptive Reward: Train neural reward model ---
            # (Note: transition collection requires tracking; for now we use
            #  reward_trace as a proxy. Full transition tracking can be added.)

            # Periodic model save
            self._campaigns_completed += 1
            if (
                state.treatment.startswith("rl_")
                and self._campaigns_completed % self.config.save_frequency == 0
            ):
                try:
                    rl_selector = self._get_rl_selector()
                    if rl_selector is not None:
                        rl_selector.save(self.config.model_path)
                        logger.info(
                            "Auto-saved RL model after %d campaigns",
                            self._campaigns_completed,
                        )
                except Exception:
                    logger.debug("Failed to auto-save RL model", exc_info=True)

        except Exception:
            logger.debug(
                "on_campaign_complete failed for campaign=%s",
                campaign_id, exc_info=True,
            )
        finally:
            # Clean up per-campaign state
            self._campaign_states.pop(campaign_id, None)

    # ------------------------------------------------------------------
    # Internal: RL selector management
    # ------------------------------------------------------------------

    def _get_rl_selector(self) -> Any:
        """Lazy-initialize and return the RL selector for the configured backend."""
        if self._rl_selector is not None and self._initialized_backend == self.config.rl_backend:
            return self._rl_selector

        backend = self.config.rl_backend

        if backend == "dqn":
            from app.services.dqn_strategy_selector import DQNStrategySelector, DQNConfig
            config = DQNConfig(model_save_path=self.config.model_path)
            self._rl_selector = DQNStrategySelector(config)

            # Try to load existing model
            try:
                from pathlib import Path
                if Path(self.config.model_path).exists():
                    self._rl_selector.load(self.config.model_path)
                    logger.info("Loaded existing DQN model from %s", self.config.model_path)
            except Exception:
                logger.debug("No existing DQN model found, starting fresh", exc_info=True)

        elif backend == "q_learning":
            from app.services.rl_strategy_selector import RLStrategySelector
            self._rl_selector = RLStrategySelector()

            # Q-learning selector uses different save/load API
            try:
                from pathlib import Path
                model_path = Path("models/rl_q_table.pkl")
                if model_path.exists():
                    self._rl_selector.load(model_path)
                    logger.info("Loaded existing Q-learning model")
            except Exception:
                logger.debug("No existing Q-learning model found, starting fresh", exc_info=True)

        elif backend == "ppo":
            try:
                from app.services.ppo_strategy_selector import PPOStrategySelector, PPOConfig
                ppo_config = PPOConfig(model_save_path=self.config.model_path.replace("dqn", "ppo"))
                self._rl_selector = PPOStrategySelector(ppo_config)

                # Try to load existing model
                try:
                    from pathlib import Path as _Path
                    if _Path(ppo_config.model_save_path).exists():
                        self._rl_selector.load(ppo_config.model_save_path)
                        logger.info("Loaded existing PPO model from %s", ppo_config.model_save_path)
                except Exception:
                    logger.debug("No existing PPO model found, starting fresh", exc_info=True)

                logger.info("Loaded PPO strategy selector")
            except ImportError:
                logger.warning(
                    "PPO backend requested but ppo_strategy_selector not available, "
                    "falling back to DQN"
                )
                from app.services.dqn_strategy_selector import DQNStrategySelector, DQNConfig
                config = DQNConfig(model_save_path=self.config.model_path)
                self._rl_selector = DQNStrategySelector(config)

        else:
            raise ValueError(f"Unknown rl_backend: {backend}")

        self._initialized_backend = backend
        return self._rl_selector

    def _should_use_rl(self, campaign_id: str) -> bool:
        """Determine whether to use RL for this campaign."""
        mode = self.config.mode

        if mode == "rule_based":
            return False
        elif mode == "rl":
            return True
        elif mode == "ab_test":
            # Deterministic assignment via hash
            return self._ab_test_assignment(campaign_id) == "rl"
        else:
            logger.warning("Unknown router mode: %s, defaulting to rule_based", mode)
            return False

    def _ab_test_assignment(self, campaign_id: str) -> str:
        """Deterministic A/B test assignment based on campaign_id hash.

        Uses SHA-256 hash modulo 100 for uniform distribution.
        Same campaign_id always gets the same assignment.
        """
        hash_val = int(hashlib.sha256(campaign_id.encode()).hexdigest(), 16)
        bucket = hash_val % 100
        if bucket < self.config.ab_test_rl_fraction * 100:
            return "rl"
        return "rule_based"

    # ------------------------------------------------------------------
    # Internal: Build RL-compatible StrategyDecision
    # ------------------------------------------------------------------

    def _build_rl_decision(
        self,
        action_id: int,
        backend_name: str,
        snapshot: Any,
        diagnostics: Any,
    ) -> Any:
        """Build a StrategyDecision from RL action output.

        Maps RL actions (explore/exploit/refine/stabilize) to the
        same StrategyDecision format the orchestrator expects.
        """
        from app.services.strategy_selector import StrategyDecision
        from app.services.rl_strategy_selector import ACTIONS, ACTION_TO_BACKEND

        action_name = ACTIONS[action_id]

        # Map RL action to backend, respecting available backends
        available = snapshot.available_backends or []
        resolved_backend = ACTION_TO_BACKEND.get(action_name, "built_in")

        # If the resolved backend isn't available, fall back
        if available and resolved_backend not in available:
            resolved_backend = available[0] if available else "built_in"

        # Phase label from action name
        phase_map = {
            "explore": "exploration",
            "exploit": "exploitation",
            "refine": "refinement",
            "stabilize": "stabilization",
        }
        phase = phase_map.get(action_name, "exploitation")

        return StrategyDecision(
            backend_name=resolved_backend,
            phase=phase,
            reason=f"[RL-{self.config.rl_backend}] Selected {action_name} "
                   f"(action={action_id}) → backend={resolved_backend}",
            confidence=0.5,  # Default; overridden by confidence gate
            fallback_backend="built_in",
            diagnostics=diagnostics,
            explanation=f"RL agent ({self.config.rl_backend}) chose {action_name} strategy",
        )

    # ------------------------------------------------------------------
    # Internal: A/B test logging
    # ------------------------------------------------------------------

    def _log_ab_result(self, campaign_id: str, state: _CampaignRLState) -> None:
        """Log campaign result for A/B test analysis."""
        try:
            if self._ab_logger is None:
                from app.services.ab_test_logger import ABTestLogger
                self._ab_logger = ABTestLogger()

            from app.services.ab_test_logger import ABTestRecord

            total_reward = sum(state.reward_trace) if state.reward_trace else 0.0

            record = ABTestRecord(
                campaign_id=campaign_id,
                treatment=state.treatment,
                n_rounds=len(state.action_history),
                final_kpi=state.prev_best_kpi,
                converged=total_reward > 0,  # Simple heuristic
                target_reached=total_reward > 1.0,  # Simple heuristic
                total_runs=len(state.action_history),
            )
            self._ab_logger.log_result(record)

        except Exception:
            logger.debug("Failed to log A/B test result", exc_info=True)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_ab_summary(self) -> dict:
        """Get A/B test summary statistics."""
        if self._ab_logger is None:
            try:
                from app.services.ab_test_logger import ABTestLogger
                self._ab_logger = ABTestLogger()
            except Exception:
                return {}

        try:
            return self._ab_logger.compute_summary()
        except Exception:
            logger.debug("Failed to compute A/B summary", exc_info=True)
            return {}

    def save_model(self) -> None:
        """Explicitly save the current RL model."""
        if self._rl_selector is not None:
            try:
                self._rl_selector.save(self.config.model_path)
                logger.info("Saved RL model to %s", self.config.model_path)
            except Exception:
                logger.debug("Failed to save RL model", exc_info=True)
