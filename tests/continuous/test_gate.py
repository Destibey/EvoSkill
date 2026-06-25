"""Tests for the quality gate: replay buffer, surrogate evaluator, policy."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.continuous.candidates import Candidate
from src.continuous.gate import (
    GateTask,
    SurrogateEvaluator,
    build_replay_buffer,
    build_surrogate_query,
    record_gate_verdict,
    run_gate,
)

from .conftest import make_episode


def _candidate(episode_ids=("e1", "e2")) -> Candidate:
    return Candidate(
        candidate_id="cand-1",
        skill_name="preserve-units",
        skill_markdown="---\nname: preserve-units\ndescription: keep units\n---\nAlways include units.",
        episode_ids=list(episode_ids),
        cluster_size=len(episode_ids),
        target_pattern="units",
    )


class _FakeVerifier:
    def __init__(self, *, score=0.9, verdict=True, raise_exc=False, no_output=False):
        self.score = score
        self.verdict = verdict
        self.raise_exc = raise_exc
        self.no_output = no_output
        self.last_query = None

    async def run(self, query):
        self.last_query = query
        if self.raise_exc:
            raise RuntimeError("verifier down")
        if self.no_output:
            return SimpleNamespace(output=None, model="m")
        return SimpleNamespace(
            output=SimpleNamespace(score=self.score, verdict=self.verdict,
                                   assertions=["generalizes"], reasoning="ok"),
            model="m",
        )


class TestReplayBuffer:
    def test_excludes_candidate_episodes(self):
        eps = [make_episode(f"e{i}", f"t{i}") for i in range(5)]
        buf = build_replay_buffer(eps, _candidate(("e1", "e2")), size=10)
        assert [e.episode_id for e in buf] == ["e0", "e3", "e4"]

    def test_size_cap(self):
        eps = [make_episode(f"x{i}", f"t{i}") for i in range(20)]
        buf = build_replay_buffer(eps, _candidate(()), size=5)
        assert len(buf) == 5

    def test_empty_when_all_excluded(self):
        eps = [make_episode("e1", "t"), make_episode("e2", "t")]
        assert build_replay_buffer(eps, _candidate(("e1", "e2")), size=10) == []


class TestSurrogateQuery:
    def test_includes_skill_and_tasks_without_answers(self):
        q = build_surrogate_query(_candidate(), [GateTask("held out task", "x")])
        assert "preserve-units" in q
        assert "held out task" in q

    def test_no_tasks_handled(self):
        q = build_surrogate_query(_candidate(), [])
        assert "no independent replay evidence" in q

    def test_caps_and_truncates(self):
        tasks = [GateTask("t" * 1000, str(i)) for i in range(20)]
        q = build_surrogate_query(_candidate(), tasks, max_tasks=3, task_chars=50)
        assert "more not shown" in q
        assert "…" in q


class TestSurrogateEvaluator:
    def test_pass(self):
        ev = SurrogateEvaluator(_FakeVerifier(score=0.9, verdict=True))
        out = asyncio.run(ev.evaluate(_candidate(), [GateTask("t")]))
        assert out.verdict is True
        assert out.score == 0.9
        assert out.assertions == ["generalizes"]

    def test_score_clamped(self):
        out = asyncio.run(SurrogateEvaluator(_FakeVerifier(score=5.0)).evaluate(_candidate(), []))
        assert out.score == 1.0

    def test_no_output_is_fail(self):
        out = asyncio.run(SurrogateEvaluator(_FakeVerifier(no_output=True)).evaluate(_candidate(), []))
        assert out.verdict is False
        assert out.score == 0.0

    def test_verifier_exception_is_fail(self):
        out = asyncio.run(SurrogateEvaluator(_FakeVerifier(raise_exc=True)).evaluate(_candidate(), []))
        assert out.verdict is False
        assert "verifier error" in out.detail


class TestRunGate:
    def test_pass_requires_verdict_and_threshold(self):
        ev = SurrogateEvaluator(_FakeVerifier(score=0.9, verdict=True))
        v = asyncio.run(run_gate(_candidate(), [make_episode("e", "t")], ev, threshold=0.6))
        assert v.passed is True
        assert v.method == "surrogate"
        assert v.n_tasks == 1

    def test_low_score_fails(self):
        ev = SurrogateEvaluator(_FakeVerifier(score=0.4, verdict=True))
        v = asyncio.run(run_gate(_candidate(), [make_episode("e", "t")], ev, threshold=0.6))
        assert v.passed is False

    def test_verdict_false_fails_even_high_score(self):
        ev = SurrogateEvaluator(_FakeVerifier(score=0.99, verdict=False))
        v = asyncio.run(run_gate(_candidate(), [make_episode("e", "t")], ev, threshold=0.6))
        assert v.passed is False

    def test_no_replay_tasks_fails_closed(self):
        verifier = _FakeVerifier(score=0.99, verdict=True)
        ev = SurrogateEvaluator(verifier)
        v = asyncio.run(run_gate(_candidate(), [], ev, threshold=0.6))
        assert v.passed is False
        assert v.score == 0.0
        assert v.n_tasks == 0
        assert "no held-out replay tasks" in v.detail
        assert verifier.last_query is None


class TestRecordGateVerdict:
    def test_persists_audit_fields_without_status_change(self, tmp_path):
        from src.continuous.candidates import CandidateStore

        store = CandidateStore(tmp_path / "cands")
        candidate = _candidate()
        store.save(candidate)
        ev = SurrogateEvaluator(_FakeVerifier(score=0.9, verdict=True))
        verdict = asyncio.run(run_gate(candidate, [make_episode("e3", "t")], ev, threshold=0.6))

        updated = record_gate_verdict(store, candidate, verdict, evaluated_at="2026-06-25T00:00:00Z")

        saved = store.get(candidate.candidate_id)
        assert updated.status == "pending"
        assert saved.status == "pending"
        assert saved.extra["gate_passed"] is True
        assert saved.extra["gate_score"] == 0.9
        assert saved.extra["gate_n_tasks"] == 1
        assert saved.extra["gate_evaluated_at"] == "2026-06-25T00:00:00Z"
