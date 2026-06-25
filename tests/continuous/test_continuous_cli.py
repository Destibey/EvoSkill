"""CLI tests for `evoskill harvest` (dry-run) and `evoskill candidates`.

Full harvest (LLM distillation) is covered by test_harvest.py against a fake
distiller; here we exercise the click wiring, argument resolution, and the
no-LLM paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from src.cli.commands.candidates import candidates_cmd
from src.cli.commands.gate import gate_cmd
from src.cli.commands.graduate import graduate_cmd, reject_cmd
from src.cli.commands.harvest import harvest_cmd
from src.cli.commands.library import library_cmd
from src.cli.commands.skill_eval import skill_eval_cmd
from src.cli.commands.watch import watch_cmd
from src.continuous.candidates import Candidate, CandidateStore

from .conftest import make_trial


def _write_skill(skills_dir: Path, name: str, description: str) -> None:
    d = skills_dir / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n\nbody")


def _project(tmp_path: Path, continuous_toml: str = "") -> Path:
    evoskill = tmp_path / ".evoskill"
    evoskill.mkdir()
    (evoskill / "config.toml").write_text('[harness]\nname = "claude"\n' + continuous_toml)
    return evoskill / "config.toml"


class TestHarvestCli:
    def test_dry_run_collects_and_clusters(self, tmp_path):
        cfg = _project(tmp_path)
        traces = tmp_path / "traces"
        # three failing trials with identical task text → one cluster
        for i in range(3):
            make_trial(traces, f"t{i}__X{i}", reward="0", task_name=f"bench/t{i}")
        result = CliRunner().invoke(harvest_cmd, [
            "--config", str(cfg), "--traces", str(traces),
            "--source", "harbor", "--dry-run", "--min-cluster-size", "1",
        ])
        assert result.exit_code == 0, result.output
        assert "Collected 3 episode(s)" in result.output
        assert "no candidates written" in result.output

    def test_no_usable_sources_errors(self, tmp_path):
        cfg = _project(tmp_path)
        # jsonl source but no jsonl_path configured → build_readers yields nothing
        result = CliRunner().invoke(harvest_cmd, ["--config", str(cfg), "--source", "jsonl"])
        assert result.exit_code == 1
        assert "no usable trace sources" in result.output


class TestCandidatesCli:
    def test_empty(self, tmp_path):
        cfg = _project(tmp_path)
        result = CliRunner().invoke(candidates_cmd, ["--config", str(cfg)])
        assert result.exit_code == 0
        assert "No candidates yet" in result.output

    def test_list_and_show(self, tmp_path):
        cfg = _project(tmp_path)
        from src.cli.config import load_config
        store = CandidateStore(load_config(config_path=cfg).continuous_candidates_dir)
        store.save(Candidate(
            candidate_id="my-skill-abc",
            skill_name="my-skill",
            skill_markdown="---\nname: my-skill\ndescription: d\n---\nthe body rule",
            target_pattern="recurring thing",
            episode_ids=["e1", "e2", "e3"],
            cluster_size=3,
        ))

        listing = CliRunner().invoke(candidates_cmd, ["--config", str(cfg)])
        assert listing.exit_code == 0
        assert "my-skill-abc" in listing.output
        assert "1 candidate(s)" in listing.output

        shown = CliRunner().invoke(candidates_cmd, ["--config", str(cfg), "--show", "my-skill-abc"])
        assert shown.exit_code == 0
        assert "the body rule" in shown.output
        assert "recurring thing" in shown.output  # full pattern, not truncated in --show

    def test_show_missing(self, tmp_path):
        cfg = _project(tmp_path)
        result = CliRunner().invoke(candidates_cmd, ["--config", str(cfg), "--show", "nope"])
        assert result.exit_code == 1
        assert "no candidate" in result.output

    def test_status_filter(self, tmp_path):
        cfg = _project(tmp_path)
        from src.cli.config import load_config
        store = CandidateStore(load_config(config_path=cfg).continuous_candidates_dir)
        store.save(Candidate(candidate_id="a", skill_name="a", skill_markdown="x", episode_ids=["e"]))
        store.save(Candidate(candidate_id="b", skill_name="b", skill_markdown="y",
                             episode_ids=["f"], status="graduated"))
        result = CliRunner().invoke(candidates_cmd, ["--config", str(cfg), "--status", "graduated"])
        assert result.exit_code == 0
        assert "b" in result.output
        # 'a' (pending) should be filtered out of the table rows
        assert "1 candidate(s)" in result.output

    def test_lists_gate_status(self, tmp_path):
        cfg = _project(tmp_path)
        from src.cli.config import load_config
        store = CandidateStore(load_config(config_path=cfg).continuous_candidates_dir)
        store.save(Candidate(
            candidate_id="a", skill_name="a", skill_markdown="x", episode_ids=["e"],
            extra={
                "gate_passed": False,
                "gate_score": 0.25,
                "gate_improvement": 0.5,
                "gate_detail": "not enough replay",
            },
        ))

        listing = CliRunner().invoke(candidates_cmd, ["--config", str(cfg)])
        assert listing.exit_code == 0
        assert "fail 0.25 +0.50" in listing.output

        shown = CliRunner().invoke(candidates_cmd, ["--config", str(cfg), "--show", "a"])
        assert shown.exit_code == 0
        assert "gate: fail 0.25 +0.50" in shown.output
        assert "not enough replay" in shown.output


class _FakeGateAgent:
    def __init__(self, _options, _schema):
        pass

    async def run(self, query):
        return SimpleNamespace(
            output=SimpleNamespace(
                score=0.84,
                verdict=True,
                assertions=["held-out task is covered"],
                reasoning="candidate generalizes",
            ),
            model="fake",
            total_cost_usd=0.0,
        )


class _CompareGateAgent:
    def __init__(self, _options, _schema):
        pass

    async def run(self, query):
        score = 0.2 if "old rule" in query else 0.84
        return SimpleNamespace(
            output=SimpleNamespace(
                score=score,
                verdict=True,
                assertions=["candidate beats baseline"],
                reasoning="candidate improves the held-out replay",
            ),
            model="fake",
            total_cost_usd=0.0,
        )


class TestGateCli:
    def _candidate(self):
        return Candidate(
            candidate_id="units-abc", skill_name="preserve-units",
            skill_markdown="---\nname: preserve-units\ndescription: d\n---\nrule",
            episode_ids=["source"], cluster_size=1,
        )

    def _jsonl(self, tmp_path):
        trace = tmp_path / ".evoskill" / "continuous" / "complaints.jsonl"
        trace.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {"episode_id": "source", "task": "source complaint", "outcome": "failure"},
            {"episode_id": "heldout", "task": "held-out similar complaint", "outcome": "failure"},
        ]
        trace.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return trace

    def test_gate_records_verdict_without_graduating(self, tmp_path, monkeypatch):
        cfg = _project(tmp_path, '\n[continuous]\ntrace_sources = ["jsonl"]\n')
        trace = self._jsonl(tmp_path)
        from src.cli.config import load_config
        import src.harness as harness

        loaded = load_config(config_path=cfg)
        store = CandidateStore(loaded.continuous_candidates_dir)
        store.save(self._candidate())
        monkeypatch.setattr(harness, "set_sdk", lambda _name: None)
        monkeypatch.setattr(harness, "Agent", _FakeGateAgent)

        result = CliRunner().invoke(
            gate_cmd, ["--config", str(cfg), "--jsonl", str(trace), "--source", "jsonl", "units-abc"])

        assert result.exit_code == 0, result.output
        assert "PASS" in result.output
        assert "Candidate remains buffered" in result.output
        saved = store.get("units-abc")
        assert saved.status == "pending"
        assert saved.extra["gate_passed"] is True
        assert saved.extra["gate_score"] == 0.84
        assert saved.extra["gate_n_tasks"] == 1
        assert not (loaded.skills_dir / "preserve-units" / "SKILL.md").exists()

    def test_gate_compares_against_live_baseline(self, tmp_path, monkeypatch):
        cfg = _project(tmp_path, '\n[continuous]\ntrace_sources = ["jsonl"]\n')
        trace = self._jsonl(tmp_path)
        from src.cli.config import load_config
        import src.harness as harness

        loaded = load_config(config_path=cfg)
        _write_skill(loaded.skills_dir, "preserve-units", "old")
        (loaded.skills_dir / "preserve-units" / "SKILL.md").write_text(
            "---\nname: preserve-units\ndescription: old\n---\nold rule"
        )
        store = CandidateStore(loaded.continuous_candidates_dir)
        store.save(self._candidate())
        monkeypatch.setattr(harness, "set_sdk", lambda _name: None)
        monkeypatch.setattr(harness, "Agent", _CompareGateAgent)

        result = CliRunner().invoke(
            gate_cmd, ["--config", str(cfg), "--jsonl", str(trace), "--source", "jsonl", "units-abc"])

        assert result.exit_code == 0, result.output
        assert "Baseline preserve-units" in result.output
        assert "improvement=+0.64" in result.output
        saved = store.get("units-abc")
        assert saved.extra["gate_baseline_name"] == "preserve-units"
        assert saved.extra["gate_baseline_score"] == 0.2
        assert saved.extra["gate_improvement"] == 0.64

    def test_gate_fails_closed_without_held_out_replay(self, tmp_path, monkeypatch):
        cfg = _project(tmp_path, '\n[continuous]\ntrace_sources = ["jsonl"]\n')
        trace = tmp_path / ".evoskill" / "continuous" / "complaints.jsonl"
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_text(json.dumps({"episode_id": "source", "task": "source complaint"}) + "\n")
        from src.cli.config import load_config
        import src.harness as harness

        loaded = load_config(config_path=cfg)
        store = CandidateStore(loaded.continuous_candidates_dir)
        store.save(self._candidate())
        monkeypatch.setattr(harness, "set_sdk", lambda _name: None)
        monkeypatch.setattr(harness, "Agent", _FakeGateAgent)

        result = CliRunner().invoke(
            gate_cmd, ["--config", str(cfg), "--jsonl", str(trace), "--source", "jsonl", "units-abc"])

        assert result.exit_code == 1
        assert "FAIL" in result.output
        saved = store.get("units-abc")
        assert saved.status == "pending"
        assert saved.extra["gate_passed"] is False
        assert saved.extra["gate_n_tasks"] == 0


class TestSkillEvalCli:
    def _candidate(self):
        return Candidate(
            candidate_id="units-abc", skill_name="preserve-units",
            skill_markdown="---\nname: preserve-units\ndescription: d\n---\nrule",
            target_pattern="answers miss units",
            episode_ids=["source"], cluster_size=1,
        )

    def _jsonl(self, tmp_path, *, heldout=True):
        trace = tmp_path / ".evoskill" / "continuous" / "complaints.jsonl"
        trace.parent.mkdir(parents=True, exist_ok=True)
        records = [{"episode_id": "source", "task": "source complaint", "outcome": "failure"}]
        if heldout:
            records.append({"episode_id": "heldout", "task": "held-out similar complaint",
                            "outcome": "failure"})
        trace.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return trace

    def test_exports_agent_skills_eval_packet(self, tmp_path):
        cfg = _project(tmp_path, '\n[continuous]\ntrace_sources = ["jsonl"]\n')
        trace = self._jsonl(tmp_path)
        from src.cli.config import load_config

        loaded = load_config(config_path=cfg)
        store = CandidateStore(loaded.continuous_candidates_dir)
        store.save(self._candidate())
        result = CliRunner().invoke(
            skill_eval_cmd,
            ["--config", str(cfg), "--jsonl", str(trace), "--source", "jsonl", "units-abc"],
        )

        assert result.exit_code == 0, result.output
        out = loaded.project_root / ".evoskill" / "continuous" / "external-evals" / "units-abc"
        assert (out / "skills" / "preserve-units" / "SKILL.md").is_file()
        payload = json.loads((out / "skills" / "preserve-units" / "evals" / "evals.json").read_text())
        assert payload["skill_name"] == "preserve-units"
        assert payload["evals"][0]["prompt"] == "held-out similar complaint"
        assert "answers miss units" in payload["evals"][0]["expected_output"]
        config_text = (out / "agent-skills-eval.yaml").read_text()
        assert "baseline: true" in config_text
        assert "root: ./skills" in config_text
        provenance = json.loads((out / "evoskill-provenance.json").read_text())
        assert provenance["candidate_id"] == "units-abc"
        assert provenance["replay_episode_ids"] == ["heldout"]
        assert not (loaded.skills_dir / "preserve-units" / "SKILL.md").exists()

    def test_export_requires_held_out_replay(self, tmp_path):
        cfg = _project(tmp_path, '\n[continuous]\ntrace_sources = ["jsonl"]\n')
        trace = self._jsonl(tmp_path, heldout=False)
        from src.cli.config import load_config

        loaded = load_config(config_path=cfg)
        CandidateStore(loaded.continuous_candidates_dir).save(self._candidate())
        result = CliRunner().invoke(
            skill_eval_cmd,
            ["--config", str(cfg), "--jsonl", str(trace), "--source", "jsonl", "units-abc"],
        )

        assert result.exit_code == 1
        assert "no held-out replay tasks" in result.output
        out = loaded.project_root / ".evoskill" / "continuous" / "external-evals" / "units-abc"
        assert not out.exists()

    def test_missing_candidate(self, tmp_path):
        cfg = _project(tmp_path)
        result = CliRunner().invoke(skill_eval_cmd, ["--config", str(cfg), "nope"])
        assert result.exit_code == 1
        assert "no candidate" in result.output


class TestLibraryCli:
    # Force lexical similarity so tests never hit a real embedding API.
    # Lower dedupe threshold to match lexical cosine scores (~0.78 for paraphrases).
    LEXICAL = '\n[continuous.lifecycle]\nsimilarity_backend = "lexical"\ndedupe_similarity = 0.6\n'

    def test_empty(self, tmp_path):
        cfg = _project(tmp_path)
        result = CliRunner().invoke(library_cmd, ["--config", str(cfg)])
        assert result.exit_code == 0
        assert "No skills yet" in result.output

    def test_list_with_stats(self, tmp_path):
        cfg = _project(tmp_path)
        _write_skill(tmp_path / ".claude" / "skills", "preserve-units", "include units")
        result = CliRunner().invoke(library_cmd, ["--config", str(cfg)])
        assert result.exit_code == 0
        assert "preserve-units" in result.output
        assert "1 skill(s)" in result.output

    def test_duplicates(self, tmp_path):
        cfg = _project(tmp_path, self.LEXICAL)
        sd = tmp_path / ".claude" / "skills"
        _write_skill(sd, "preserve-units", "always include measurement units in answers")
        _write_skill(sd, "keep-units", "always include measurement units in answers please")
        _write_skill(sd, "read-tables", "extract figures from financial tables")
        result = CliRunner().invoke(library_cmd, ["--config", str(cfg), "--duplicates"])
        assert result.exit_code == 0
        # the two near-identical unit skills should be flagged similar
        assert "preserve-units" in result.output and "keep-units" in result.output

    def test_select(self, tmp_path):
        cfg = _project(tmp_path, self.LEXICAL)
        sd = tmp_path / ".claude" / "skills"
        _write_skill(sd, "preserve-units", "include measurement units in numeric answers")
        _write_skill(sd, "read-tables", "extract figures from financial tables")
        result = CliRunner().invoke(
            library_cmd, ["--config", str(cfg), "--select", "what units for this revenue value"])
        assert result.exit_code == 0
        assert "preserve-units" in result.output

    def test_deprecated_empty(self, tmp_path):
        cfg = _project(tmp_path)
        result = CliRunner().invoke(library_cmd, ["--config", str(cfg), "--deprecated"])
        assert result.exit_code == 0
        assert "No deprecated skills" in result.output


class TestGraduateCli:
    def _candidate(self):
        return Candidate(
            candidate_id="units-abc", skill_name="preserve-units",
            skill_markdown="---\nname: preserve-units\ndescription: d\n---\nrule",
            episode_ids=["e1"], cluster_size=1,
        )

    def test_force_no_branch_refuses_without_snapshot(self, tmp_path):
        cfg = _project(tmp_path)
        from src.cli.config import load_config
        loaded = load_config(config_path=cfg)
        CandidateStore(loaded.continuous_candidates_dir).save(self._candidate())
        # --force skips the gate, but live writeback still requires a program/* snapshot.
        result = CliRunner().invoke(
            graduate_cmd, ["--config", str(cfg), "--force", "--no-branch", "units-abc"])
        assert result.exit_code == 1
        assert "refusing to graduate without a program/* snapshot" in result.output
        assert not (loaded.skills_dir / "preserve-units" / "SKILL.md").exists()
        assert CandidateStore(loaded.continuous_candidates_dir).get("units-abc").status == "pending"

    def test_graduate_missing_candidate(self, tmp_path):
        cfg = _project(tmp_path)
        result = CliRunner().invoke(graduate_cmd, ["--config", str(cfg), "--force", "nope"])
        assert result.exit_code == 1
        assert "no candidate" in result.output

    def test_reject(self, tmp_path):
        cfg = _project(tmp_path)
        from src.cli.config import load_config
        store = CandidateStore(load_config(config_path=cfg).continuous_candidates_dir)
        store.save(self._candidate())
        result = CliRunner().invoke(reject_cmd, ["--config", str(cfg), "units-abc"])
        assert result.exit_code == 0
        assert store.get("units-abc").status == "rejected"

    def test_reject_missing(self, tmp_path):
        cfg = _project(tmp_path)
        result = CliRunner().invoke(reject_cmd, ["--config", str(cfg), "nope"])
        assert result.exit_code == 1


class TestWatchCli:
    def test_once_review_no_traces(self, tmp_path):
        # review mode + empty traces dir → one tick, 0 episodes, no LLM/git needed.
        cfg = _project(tmp_path)
        (tmp_path / ".evoskill" / "harbor_jobs").mkdir(parents=True)
        result = CliRunner().invoke(watch_cmd, ["--config", str(cfg), "--once"])
        assert result.exit_code == 0, result.output
        assert "Ran 1 tick(s)" in result.output
        assert "0 episodes" in result.output

    def test_no_usable_sources_errors(self, tmp_path):
        cfg = _project(tmp_path, '\n[continuous]\ntrace_sources = ["jsonl"]\n')
        result = CliRunner().invoke(watch_cmd, ["--config", str(cfg), "--once"])
        assert result.exit_code == 1
        assert "no usable trace sources" in result.output
