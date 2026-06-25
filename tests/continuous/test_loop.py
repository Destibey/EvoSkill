"""Tests for the continuous tick and the watch loop."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.continuous.candidates import CandidateStore
from src.continuous.collector import TraceCursor
from src.continuous.episode import Outcome
from src.continuous.loop import (
    CostMeter,
    MeteredAgent,
    TickConfig,
    run_tick,
    run_watch_loop,
)
from src.continuous.skill_stats import SkillStatsStore

from .conftest import make_episode


class _StaticReader:
    source = "static"

    def __init__(self, episodes):
        self._episodes = episodes

    def read_all(self):
        return iter(self._episodes)


def _similar_failures(n, text="compute the federal deficit fiscal quarter", prefix="d"):
    return [make_episode(f"{prefix}-{i}", text) for i in range(n)]


class _FakeDistiller:
    def __init__(self, *, cost=0.01):
        self.cost = cost
        self.calls = 0

    async def run(self, query):
        self.calls += 1
        return SimpleNamespace(
            output=SimpleNamespace(
                skill_name=f"skill-{self.calls}",
                candidate_skill=f"---\nname: skill-{self.calls}\ndescription: d\n---\nrule",
                target_pattern="deficit", reasoning="r"),
            model="m", total_cost_usd=self.cost,
        )


class _FakeVerifier:
    def __init__(self, *, passing=True, cost=0.0):
        self.passing = passing
        self.cost = cost

    async def run(self, query):
        return SimpleNamespace(
            output=SimpleNamespace(score=0.9 if self.passing else 0.3, verdict=self.passing,
                                   assertions=["a"], reasoning="ok"),
            model="m", total_cost_usd=self.cost,
        )


class _FakeManager:
    def __init__(self):
        self.created = []

    def get_current(self):
        from src.registry.models import ProgramConfig
        return ProgramConfig(name="base", system_prompt={"type": "preset", "preset": "claude_code"})

    def create_program(self, name, config, parent=None):
        self.created.append((name, parent, config.metadata))
        return f"program/{name}"


def _tick(tmp_path, **kw):
    """Helper: run a tick with sensible temp stores."""
    store = CandidateStore(tmp_path / "cands")
    cursor = kw.pop("cursor", None) or TraceCursor(tmp_path / "cursor.json")
    return asyncio.run(run_tick(store=store, cursor=cursor, **kw)), store, cursor


# ── CostMeter / MeteredAgent ──────────────────────────────────────────────────


class TestCostMeter:
    def test_accumulates_and_threshold(self):
        m = CostMeter()
        m.add(0.03)
        m.add(0.04)
        assert abs(m.spent - 0.07) < 1e-9
        assert m.exceeded(0.05) is True
        assert m.exceeded(0.10) is False
        assert m.exceeded(None) is False
        assert m.exceeded(0.0) is False  # 0 = unlimited

    def test_metered_agent_records_cost(self):
        meter = CostMeter()

        class A:
            async def run(self, q):
                return SimpleNamespace(output="x", total_cost_usd=0.05)

        trace = asyncio.run(MeteredAgent(A(), meter).run("q"))
        assert trace.output == "x"
        assert meter.spent == 0.05


# ── review mode ──────────────────────────────────────────────────────────────


class TestReviewMode:
    def test_discovers_without_graduating(self, tmp_path):
        cfg = TickConfig(mode="review", min_cluster_size=3, max_candidates=2)
        d = _FakeDistiller()
        report, store, _ = _tick(
            tmp_path, readers=[_StaticReader(_similar_failures(5))], distiller=d,
            skills_dir=tmp_path / "skills", config=cfg,
        )
        assert report.episodes_collected == 5
        assert report.num_candidates == 1
        assert report.num_graduated == 0
        assert report.cost_usd == 0.01  # one distill call
        assert len(store.list()) == 1

    def test_watermark_advances(self, tmp_path):
        cfg = TickConfig(mode="review", min_cluster_size=3)
        cursor = TraceCursor(tmp_path / "cursor.json")
        eps = _similar_failures(4)
        _tick(tmp_path, readers=[_StaticReader(eps)], distiller=_FakeDistiller(),
              cursor=cursor, config=cfg)
        report2, _, _ = _tick(tmp_path, readers=[_StaticReader(eps)], distiller=_FakeDistiller(),
                              cursor=cursor, config=cfg)
        assert report2.episodes_collected == 0  # all watermarked


# ── auto mode ────────────────────────────────────────────────────────────────


class TestAutoMode:
    def test_gates_and_graduates(self, tmp_path):
        cfg = TickConfig(mode="auto", min_cluster_size=3, graduation_threshold=0.6, max_graduations=5)
        eps = _similar_failures(5) + [make_episode("heldout", "similar held-out task", outcome=Outcome.SUCCESS)]
        report, store, _ = _tick(
            tmp_path, readers=[_StaticReader(eps)], distiller=_FakeDistiller(),
            verifier=_FakeVerifier(passing=True), skills_dir=tmp_path / "skills",
            manager=_FakeManager(), config=cfg,
        )
        assert report.num_candidates == 1
        assert report.num_graduated == 1
        assert (tmp_path / "skills" / "skill-1" / "SKILL.md").is_file()
        cand_id = report.candidates[0].candidate_id
        assert store.get(cand_id).status == "graduated"
        assert report.gated[cand_id].passed is True

    def test_gate_failure_blocks_graduation(self, tmp_path):
        cfg = TickConfig(mode="auto", min_cluster_size=3)
        eps = _similar_failures(5) + [make_episode("heldout", "similar held-out task", outcome=Outcome.SUCCESS)]
        report, store, _ = _tick(
            tmp_path, readers=[_StaticReader(eps)], distiller=_FakeDistiller(),
            verifier=_FakeVerifier(passing=False), skills_dir=tmp_path / "skills",
            manager=_FakeManager(), config=cfg,
        )
        assert report.num_graduated == 0
        cand_id = report.candidates[0].candidate_id
        assert store.get(cand_id).status == "pending"
        assert store.get(cand_id).extra["gate_passed"] is False  # verdict recorded

    def test_rate_limit(self, tmp_path):
        # 3 distinct clusters → 3 candidates, but max_graduations=2
        eps = (_similar_failures(4, "deficit budget fiscal quarter total", "a")
               + _similar_failures(4, "customs duty receipts revenue bulletin", "b")
               + _similar_failures(4, "treasury bond maturity schedule listing", "c"))
        cfg = TickConfig(mode="auto", min_cluster_size=3, max_graduations=2)
        report, _, _ = _tick(
            tmp_path, readers=[_StaticReader(eps)], distiller=_FakeDistiller(),
            verifier=_FakeVerifier(passing=True), skills_dir=tmp_path / "skills",
            manager=_FakeManager(), config=cfg,
        )
        assert report.num_graduated == 2
        assert report.stopped_reason == "max_graduations"

    def test_cost_ceiling_stops(self, tmp_path):
        eps = (_similar_failures(4, "deficit budget fiscal quarter", "a")
               + _similar_failures(4, "customs duty receipts revenue", "b"))
        # distill 2 × 0.01 = 0.02; gate 0.10 each → ceiling 0.05 trips after first graduation
        cfg = TickConfig(mode="auto", min_cluster_size=3, max_graduations=99, cost_ceiling=0.05)
        report, _, _ = _tick(
            tmp_path, readers=[_StaticReader(eps)], distiller=_FakeDistiller(cost=0.01),
            verifier=_FakeVerifier(passing=True, cost=0.10), skills_dir=tmp_path / "skills",
            manager=_FakeManager(), config=cfg,
        )
        assert report.stopped_reason == "cost_ceiling"
        assert report.num_graduated < 2

    def test_no_verifier_stops(self, tmp_path):
        cfg = TickConfig(mode="auto", min_cluster_size=3)
        report, _, _ = _tick(
            tmp_path, readers=[_StaticReader(_similar_failures(4))], distiller=_FakeDistiller(),
            verifier=None, skills_dir=tmp_path / "skills", config=cfg,
        )
        assert report.stopped_reason == "no verifier for auto mode"
        assert report.num_graduated == 0

    def test_no_manager_stops_before_gate_and_graduation(self, tmp_path):
        cfg = TickConfig(mode="auto", min_cluster_size=3)
        verifier = _FakeVerifier(passing=True)
        report, store, _ = _tick(
            tmp_path, readers=[_StaticReader(_similar_failures(4))], distiller=_FakeDistiller(),
            verifier=verifier, skills_dir=tmp_path / "skills", manager=None, config=cfg,
        )
        assert report.stopped_reason == "no program manager for auto graduation"
        assert report.gated == {}
        assert report.num_graduated == 0
        saved = store.list()
        assert len(saved) == 1
        assert saved[0].status == "pending"
        assert not (tmp_path / "skills" / "skill-1" / "SKILL.md").exists()

    def test_dedup_guard_skips_existing(self, tmp_path):
        # An existing live skill the candidate duplicates.
        sd = tmp_path / "skills"
        (sd / "skill-1").mkdir(parents=True)
        (sd / "skill-1" / "SKILL.md").write_text("---\nname: skill-1\ndescription: deficit\n---\nx")

        class DupBackend:
            name = "dup"
            def rank(self, query, candidates):
                return [(0, 0.99)]  # always "duplicate"
            def pairwise(self, texts):
                return [[1.0]]

        cfg = TickConfig(mode="auto", min_cluster_size=3, dedupe_similarity=0.88)
        report, _, _ = _tick(
            tmp_path, readers=[_StaticReader(_similar_failures(4))], distiller=_FakeDistiller(),
            verifier=_FakeVerifier(passing=True), skills_dir=sd,
            similarity_backend=DupBackend(), manager=_FakeManager(), config=cfg,
        )
        assert len(report.skipped_duplicates) == 1
        assert report.num_graduated == 0


# ── credit + deprecation ─────────────────────────────────────────────────────


class TestCreditAndDeprecation:
    def test_credit_assigned(self, tmp_path):
        eps = _similar_failures(3)
        for e in eps:
            e.skills_active = ["some-skill"]
            e.outcome = Outcome.SUCCESS  # success so credit accrues
        stats = SkillStatsStore(tmp_path / "stats.json")
        cfg = TickConfig(mode="review", min_cluster_size=99)  # no clustering needed
        _tick(tmp_path, readers=[_StaticReader(eps)], distiller=_FakeDistiller(),
              stats_store=stats, skills_dir=tmp_path / "skills", config=cfg)
        assert stats.get("some-skill").episodes_active == 3

    def test_deprecation_reported(self, tmp_path):
        sd = tmp_path / "skills"
        (sd / "dead").mkdir(parents=True)
        (sd / "dead" / "SKILL.md").write_text("---\nname: dead\ndescription: d\n---\nx")
        stats = SkillStatsStore(tmp_path / "stats.json")
        s = stats.get("dead")
        s.episodes_active = 5  # active, zero contribution
        s.deprecation_strikes = 2
        stats.upsert(s)
        cfg = TickConfig(mode="review", min_cluster_size=99,
                         deprecation_baseline=0.0, deprecation_strikes=3)
        report, _, _ = _tick(tmp_path, readers=[_StaticReader([])], distiller=_FakeDistiller(),
                             stats_store=stats, skills_dir=sd, config=cfg)
        assert report.deprecation is not None
        assert "dead" in report.deprecation.candidates  # crossed the strike limit

    def test_auto_deprecate_archives(self, tmp_path):
        sd = tmp_path / "skills"
        (sd / "dead").mkdir(parents=True)
        (sd / "dead" / "SKILL.md").write_text("---\nname: dead\ndescription: d\n---\nx")
        stats = SkillStatsStore(tmp_path / "stats.json")
        s = stats.get("dead")
        s.episodes_active = 5
        s.deprecation_strikes = 2  # one more strike crosses the limit
        stats.upsert(s)
        archive = tmp_path / "deprecated"
        # auto mode + auto_deprecate + no episodes (so no candidates/gating needed)
        cfg = TickConfig(mode="auto", min_cluster_size=99, deprecation_strikes=3,
                         auto_deprecate=True)
        report, _, _ = _tick(
            tmp_path, readers=[_StaticReader([])], distiller=_FakeDistiller(),
            verifier=_FakeVerifier(), stats_store=stats, skills_dir=sd,
            archive_dir=archive, config=cfg,
        )
        assert "dead" in report.archived
        assert (archive / "dead" / "SKILL.md").is_file()
        assert not (sd / "dead").exists()


# ── watch loop ───────────────────────────────────────────────────────────────


class TestWatchLoop:
    def test_max_ticks(self):
        calls = {"n": 0}
        sleeps = []

        def tick():
            calls["n"] += 1
            return SimpleNamespace()

        n = run_watch_loop(tick, max_ticks=3, interval_sec=5, sleep=sleeps.append)
        assert n == 3
        assert calls["n"] == 3
        assert sleeps == [5, 5]  # slept between ticks, not after the last

    def test_once(self):
        calls = {"n": 0}
        n = run_watch_loop(lambda: calls.__setitem__("n", calls["n"] + 1),
                           once=True, sleep=lambda s: None)
        assert n == 1

    def test_keyboard_interrupt_stops_cleanly(self):
        def tick():
            raise KeyboardInterrupt
        n = run_watch_loop(tick, max_ticks=5, sleep=lambda s: None)
        assert n == 0
