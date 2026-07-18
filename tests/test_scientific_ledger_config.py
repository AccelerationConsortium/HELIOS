from __future__ import annotations

from app.core.config import Settings


def test_scientific_ledger_defaults_to_markdown_on_git_off(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SCIENTIFIC_LEDGER_ROOT", raising=False)
    monkeypatch.delenv("SCIENTIFIC_LEDGER_ENABLED", raising=False)
    monkeypatch.delenv("SCIENTIFIC_LEDGER_GIT_ENABLED", raising=False)
    settings = Settings()
    assert settings.scientific_ledger_root == tmp_path / "data" / "scientific_ledger"
    assert settings.scientific_ledger_enabled is True
    assert settings.scientific_ledger_git_enabled is False
    assert settings.scientific_ledger_git_auto_init is True


def test_scientific_ledger_settings_are_configurable(monkeypatch, tmp_path):
    root = tmp_path / "campaign-ledger"
    monkeypatch.setenv("SCIENTIFIC_LEDGER_ROOT", str(root))
    monkeypatch.setenv("SCIENTIFIC_LEDGER_ENABLED", "false")
    monkeypatch.setenv("SCIENTIFIC_LEDGER_GIT_ENABLED", "true")
    monkeypatch.setenv("SCIENTIFIC_LEDGER_GIT_AUTO_INIT", "false")
    settings = Settings()
    assert settings.scientific_ledger_root == root
    assert settings.scientific_ledger_enabled is False
    assert settings.scientific_ledger_git_enabled is True
    assert settings.scientific_ledger_git_auto_init is False
