"""Tests for the [continuous] config section and derived paths."""

from __future__ import annotations

from pathlib import Path

from src.cli.config import ContinuousConfig, load_config


def _write_project(tmp_path: Path, continuous_toml: str = "") -> Path:
    evoskill = tmp_path / ".evoskill"
    evoskill.mkdir()
    (evoskill / "config.toml").write_text('[harness]\nname = "claude"\n' + continuous_toml)
    return evoskill / "config.toml"


class TestContinuousConfigDefaults:
    def test_defaults(self):
        c = ContinuousConfig()
        assert c.enabled is False
        assert c.trace_sources == ["harbor"]
        assert c.focus == "failure"
        assert c.min_cluster_size == 3
        assert c.success_threshold == 1.0

    def test_absent_section_uses_defaults(self, tmp_path):
        cfg = load_config(config_path=_write_project(tmp_path))
        assert cfg.continuous.enabled is False
        assert cfg.continuous.trace_sources == ["harbor"]


class TestContinuousConfigParsing:
    def test_overrides_parsed(self, tmp_path):
        toml = (
            "\n[continuous]\n"
            "enabled = true\n"
            'trace_sources = ["harbor", "jsonl"]\n'
            "harvest_window = 50\n"
            "min_cluster_size = 5\n"
            "similarity_threshold = 0.5\n"
            'focus = "both"\n'
        )
        cfg = load_config(config_path=_write_project(tmp_path, toml))
        assert cfg.continuous.enabled is True
        assert cfg.continuous.trace_sources == ["harbor", "jsonl"]
        assert cfg.continuous.harvest_window == 50
        assert cfg.continuous.min_cluster_size == 5
        assert cfg.continuous.similarity_threshold == 0.5
        assert cfg.continuous.focus == "both"


class TestLifecycleConfig:
    def test_defaults(self, tmp_path):
        cfg = load_config(config_path=_write_project(tmp_path))
        lc = cfg.continuous.lifecycle
        assert lc.similarity_backend == "embedding"
        assert lc.embedding_provider == "openai"
        assert lc.embedding_model == "text-embedding-3-small"
        assert lc.dedupe_similarity == 0.88
        assert lc.deprecation_strikes == 3
        assert lc.retrieval_top_k == 6

    def test_nested_section_parsed(self, tmp_path):
        toml = (
            "\n[continuous]\nenabled = true\n"
            "\n[continuous.lifecycle]\n"
            'similarity_backend = "lexical"\n'
            "dedupe_similarity = 0.75\n"
            "retrieval_top_k = 10\n"
        )
        cfg = load_config(config_path=_write_project(tmp_path, toml))
        assert cfg.continuous.enabled is True
        assert cfg.continuous.lifecycle.similarity_backend == "lexical"
        assert cfg.continuous.lifecycle.dedupe_similarity == 0.75
        assert cfg.continuous.lifecycle.retrieval_top_k == 10


class TestGraduationConfig:
    def test_defaults(self, tmp_path):
        cfg = load_config(config_path=_write_project(tmp_path))
        assert cfg.continuous.graduation_mode == "review"
        assert cfg.continuous.graduation_threshold == 0.6
        assert cfg.continuous.shadow_eval_size == 10
        assert cfg.continuous.max_graduations_per_window == 2

    def test_overrides(self, tmp_path):
        toml = (
            "\n[continuous]\n"
            'graduation_mode = "auto"\n'
            "graduation_threshold = 0.8\n"
            "shadow_eval_size = 25\n"
        )
        cfg = load_config(config_path=_write_project(tmp_path, toml))
        assert cfg.continuous.graduation_mode == "auto"
        assert cfg.continuous.graduation_threshold == 0.8
        assert cfg.continuous.shadow_eval_size == 25


class TestWatchConfig:
    def test_defaults(self, tmp_path):
        cfg = load_config(config_path=_write_project(tmp_path))
        assert cfg.continuous.poll_interval_sec == 600
        assert cfg.continuous.cost_ceiling_usd_per_tick == 0.0
        assert cfg.continuous.auto_deprecate is False

    def test_overrides(self, tmp_path):
        toml = (
            "\n[continuous]\n"
            "poll_interval_sec = 120\n"
            "cost_ceiling_usd_per_tick = 2.5\n"
            "auto_deprecate = true\n"
        )
        cfg = load_config(config_path=_write_project(tmp_path, toml))
        assert cfg.continuous.poll_interval_sec == 120
        assert cfg.continuous.cost_ceiling_usd_per_tick == 2.5
        assert cfg.continuous.auto_deprecate is True


class TestDerivedPaths:
    def test_traces_root_defaults_to_harbor_jobs(self, tmp_path):
        cfg = load_config(config_path=_write_project(tmp_path))
        assert cfg.continuous_traces_root == tmp_path / ".evoskill" / "harbor_jobs"

    def test_jsonl_path_unset_by_default(self, tmp_path):
        cfg = load_config(config_path=_write_project(tmp_path))
        assert cfg.continuous_jsonl_path is None

    def test_jsonl_path_relative_override(self, tmp_path):
        cfg = load_config(config_path=_write_project(
            tmp_path,
            '\n[continuous]\njsonl_path = ".evoskill/continuous/complaints.jsonl"\n',
        ))
        assert cfg.continuous_jsonl_path == tmp_path / ".evoskill" / "continuous" / "complaints.jsonl"

    def test_jsonl_path_absolute_override(self, tmp_path):
        abs_path = tmp_path / "abs" / "complaints.jsonl"
        cfg = load_config(config_path=_write_project(
            tmp_path, f'\n[continuous]\njsonl_path = "{abs_path}"\n'))
        assert cfg.continuous_jsonl_path == abs_path

    def test_complaints_path_defaults_under_continuous_dir(self, tmp_path):
        cfg = load_config(config_path=_write_project(tmp_path))
        assert cfg.continuous_complaints_path == tmp_path / ".evoskill" / "continuous" / "complaints.jsonl"

    def test_traces_root_relative_override(self, tmp_path):
        cfg = load_config(config_path=_write_project(tmp_path, '\n[continuous]\ntraces_root = "mytraces"\n'))
        assert cfg.continuous_traces_root == tmp_path / "mytraces"

    def test_traces_root_absolute_override(self, tmp_path):
        abs_path = tmp_path / "abs" / "traces"
        cfg = load_config(config_path=_write_project(
            tmp_path, f'\n[continuous]\ntraces_root = "{abs_path}"\n'))
        assert cfg.continuous_traces_root == abs_path

    def test_candidates_dir(self, tmp_path):
        cfg = load_config(config_path=_write_project(tmp_path))
        assert cfg.continuous_candidates_dir == tmp_path / ".evoskill" / "continuous" / "candidates"

    def test_harbor_jobs_dir_respects_config(self, tmp_path):
        cfg = load_config(config_path=_write_project(tmp_path))
        # default
        assert cfg.harbor_jobs_dir == tmp_path / ".evoskill" / "harbor_jobs"
