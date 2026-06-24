"""CLI tests for turning operator complaints into harvestable JSONL traces."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from src.cli.commands.harvest import harvest_cmd
from src.cli.main import cli
from src.continuous.collector import JsonlReader
from src.continuous.episode import Outcome


def _project(tmp_path: Path, continuous_toml: str = "") -> Path:
    evoskill = tmp_path / ".evoskill"
    evoskill.mkdir()
    (evoskill / "config.toml").write_text('[harness]\nname = "claude"\n' + continuous_toml)
    return evoskill / "config.toml"


class TestComplaintCli:
    def test_records_failure_trace_at_default_path(self, tmp_path):
        cfg = _project(tmp_path)

        result = CliRunner().invoke(cli, [
            "complaint",
            "sync audit missed the requested boundary",
            "--config", str(cfg),
            "--skill", "sync-audit",
            "--task", "Guide Hub stage audit",
            "--timestamp", "2026-06-24T00:00:00Z",
            "--episode-id", "complaint-1",
        ])

        assert result.exit_code == 0, result.output
        assert "complaints.jsonl" in result.output

        out = tmp_path / ".evoskill" / "continuous" / "complaints.jsonl"
        records = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(records) == 1
        record = records[0]
        assert record["episode_id"] == "complaint-1"
        assert record["outcome"] == "failure"
        assert record["skills_active"] == ["sync-audit"]
        assert "Guide Hub stage audit" in record["task"]
        assert "sync audit missed the requested boundary" in record["task"]
        assert record["extra"]["kind"] == "operator_complaint"

        ep = next(JsonlReader(out).read_all())
        assert ep.outcome is Outcome.FAILURE
        assert ep.skills_active == ["sync-audit"]
        assert ep.agent_name == "operator-complaint"
        assert ep.timestamp == "2026-06-24T00:00:00Z"
        assert ep.extra["complaint"] == "sync audit missed the requested boundary"

    def test_harvest_dry_run_reads_recorded_complaint_trace(self, tmp_path):
        cfg = _project(tmp_path)

        recorded = CliRunner().invoke(cli, [
            "complaint",
            "audit evidence was too thin",
            "--config", str(cfg),
            "--skill", "blueprint-audit-loop",
            "--episode-id", "complaint-2",
        ])
        assert recorded.exit_code == 0, recorded.output

        harvested = CliRunner().invoke(harvest_cmd, [
            "--config", str(cfg),
            "--source", "jsonl",
            "--jsonl", ".evoskill/continuous/complaints.jsonl",
            "--dry-run",
            "--min-cluster-size", "1",
        ])

        assert harvested.exit_code == 0, harvested.output
        assert "Collected 1 episode(s)" in harvested.output
        assert "failure clusters" in harvested.output

    def test_custom_relative_output_is_project_relative(self, tmp_path):
        cfg = _project(tmp_path)

        result = CliRunner().invoke(cli, [
            "complaint",
            "profile did not explain dependencies",
            "--config", str(cfg),
            "--output", "custom/complaints.jsonl",
            "--episode-id", "complaint-3",
        ])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "custom" / "complaints.jsonl").is_file()
