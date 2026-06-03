"""HELIOS Agent system -- typed agents with unified interface.

Individual agents follow the BaseAgent protocol. For paper-aligned
grouping, use the four specialist swarms via SwarmFactory.

Note: this package's __init__ is intentionally empty. Eagerly importing
every agent here triggers a circular dependency through
``app.services.llm_gateway`` (which imports ``app.services.agent_context``
which imports ``app.agents.base``). Import agents from their submodules
directly (e.g. ``from app.agents.query_agent import QueryAgent``) or let
Python resolve them lazily.
"""
