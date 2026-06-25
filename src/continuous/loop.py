"""The continuous-evolution tick — one pass of the always-on loop.

`run_tick` composes the tested Phase 0–3 building blocks into a single, bounded
pass:

    COLLECT → CREDIT → CLUSTER → DISTILL → (auto: GATE → GRADUATE) → DEPRECATION → WATERMARK

It is the unit the `evoskill watch` daemon repeats. Two modes:

* **review** (default, cheap): discovery only — collect, credit, cluster, distill
  into the candidate buffer. A human gates/applies later via `evoskill graduate`.
* **auto**: the full pipeline — each fresh candidate is gated on a held-out
  replay buffer and, on pass, graduated as a `program/*` branch, subject to
  per-tick safety rails.

Safety rails (auto mode): a dedup-guard skips candidates near-duplicate to live
skills; `max_graduations` rate-limits promotions; a `CostMeter` enforces
`cost_ceiling`; deprecation is reported and only archives when `auto_deprecate`
is set. The watermark advances every tick so work is never repeated.

Everything is injected (readers, agents, stores, manager) so the whole tick is
testable with fakes — no real LLM, embeddings, or git.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .candidates import Candidate, CandidateStore
from .cluster import cluster_episodes
from .collector import TraceCollector, TraceCursor, TraceReader
from .episode import Outcome
from .gate import GateVerdict, SurrogateEvaluator, build_replay_buffer, record_gate_verdict, run_gate
from .graduation import graduate
from .harvest import distill_clusters
from .library import SkillLibrary
from .lifecycle import DeprecationReport, archive_skill, evaluate_deprecation
from .similarity import SimilarityBackend
from .skill_stats import SkillStatsStore, assign_credit_batch

_Logger = Callable[[str], None]


class CostMeter:
    """Accumulates spend across agent calls within a tick."""

    def __init__(self) -> None:
        self.spent = 0.0

    def add(self, usd: float) -> None:
        self.spent += max(0.0, usd)

    def exceeded(self, ceiling: float | None) -> bool:
        return ceiling is not None and ceiling > 0 and self.spent >= ceiling


class MeteredAgent:
    """Wrap an agent so each `run()` adds its trace cost to a shared meter.

    Duck-typed: works with any object exposing `async run(query) -> trace` (an
    `src.harness.Agent`, or a test fake), so no Phase 0–3 code changes.
    """

    def __init__(self, agent: Any, meter: CostMeter) -> None:
        self._agent = agent
        self._meter = meter

    async def run(self, query: str) -> Any:
        trace = await self._agent.run(query)
        self._meter.add(float(getattr(trace, "total_cost_usd", 0.0) or 0.0))
        return trace


@dataclass
class TickConfig:
    """Policy knobs for one tick (the collaborators are passed separately)."""

    mode: str = "review"                  # "review" | "auto"
    window: int = 200
    focus: Outcome = Outcome.FAILURE
    min_cluster_size: int = 3
    similarity_threshold: float = 0.3
    max_candidates: int | None = None
    concurrency: int = 4
    # gate / graduation (auto mode)
    graduation_threshold: float = 0.6
    shadow_eval_size: int = 10
    max_graduations: int = 2
    dedupe_similarity: float = 0.88
    # deprecation
    deprecation_baseline: float = 0.0
    deprecation_strikes: int = 3
    auto_deprecate: bool = False
    # cost
    cost_ceiling: float | None = None


@dataclass
class TickReport:
    """What one tick did."""

    episodes_collected: int = 0
    credited: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    gated: dict[str, GateVerdict] = field(default_factory=dict)
    graduated: list[str] = field(default_factory=list)
    skipped_duplicates: list[str] = field(default_factory=list)
    deprecation: DeprecationReport | None = None
    archived: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    stopped_reason: str | None = None

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    @property
    def num_graduated(self) -> int:
        return len(self.graduated)


def _candidate_text(candidate: Candidate) -> str:
    """Comparable text for the dedup guard (mirrors Skill.text shape)."""
    return f"{candidate.skill_name}: {candidate.target_pattern}".strip().rstrip(":").strip()


async def run_tick(
    *,
    readers: list[TraceReader],
    store: CandidateStore,
    distiller: Any,
    cursor: TraceCursor | None = None,
    verifier: Any | None = None,
    skills_dir: str | Path | None = None,
    stats_store: SkillStatsStore | None = None,
    similarity_backend: SimilarityBackend | None = None,
    manager: Any | None = None,
    archive_dir: str | Path | None = None,
    library: SkillLibrary | None = None,
    config: TickConfig | None = None,
    log: _Logger | None = None,
) -> TickReport:
    """Run one continuous-evolution tick. See module docstring for the flow."""
    cfg = config or TickConfig()
    emit = log or (lambda _m: None)
    meter = CostMeter()
    report = TickReport()

    # 1. COLLECT — advance the watermark so each tick sees only new episodes.
    episodes = TraceCollector(readers, cursor=cursor).collect(advance=True, limit=cfg.window)
    report.episodes_collected = len(episodes)
    emit(f"collected {len(episodes)} episode(s)")

    # 2. CREDIT — attribute outcomes to the skills that were active.
    if stats_store is not None:
        report.credited = assign_credit_batch(stats_store, episodes)

    # 3-4. CLUSTER + DISTILL
    clusters = cluster_episodes(
        episodes, focus=cfg.focus, min_cluster_size=cfg.min_cluster_size,
        similarity_threshold=cfg.similarity_threshold, max_clusters=cfg.max_candidates,
    )
    candidates, _failed, _cost = await distill_clusters(
        clusters, MeteredAgent(distiller, meter), store,
        source="watch", concurrency=cfg.concurrency, log=emit,
    )
    report.candidates = candidates
    emit(f"distilled {len(candidates)} candidate(s)")

    lib = library or (SkillLibrary(skills_dir) if skills_dir is not None else None)

    # 5/9. DEPRECATION — report strikes; archive only when explicitly enabled.
    if stats_store is not None and lib is not None:
        report.deprecation = evaluate_deprecation(
            stats_store, lib.names(),
            baseline=cfg.deprecation_baseline, strikes_limit=cfg.deprecation_strikes,
        )
        if (cfg.mode == "auto" and cfg.auto_deprecate and archive_dir is not None
                and report.deprecation.candidates):
            for name in report.deprecation.candidates:
                skill = lib.get(name)
                if skill is None:
                    continue
                try:
                    archive_skill(skills_dir, skill.dir_name, archive_dir)
                    report.archived.append(name)
                except FileNotFoundError:
                    continue

    # review mode = discovery only; a human gates/applies later.
    if cfg.mode != "auto":
        report.cost_usd = meter.spent
        return report

    # 6-8. GATE + GRADUATE (auto mode)
    if verifier is None:
        report.stopped_reason = "no verifier for auto mode"
        report.cost_usd = meter.spent
        return report
    if candidates and manager is None:
        report.stopped_reason = "no program manager for auto graduation"
        report.cost_usd = meter.spent
        return report

    existing_texts = [s.text for s in lib.list()] if lib is not None else []
    evaluator = SurrogateEvaluator(MeteredAgent(verifier, meter), max_tasks=cfg.shadow_eval_size)
    graduations = 0

    for candidate in candidates:
        if meter.exceeded(cfg.cost_ceiling):
            report.stopped_reason = "cost_ceiling"
            break
        if graduations >= cfg.max_graduations:
            report.stopped_reason = "max_graduations"
            break

        # Dedup guard: don't graduate something the library already covers.
        if similarity_backend is not None and existing_texts:
            ranked = similarity_backend.rank(_candidate_text(candidate), existing_texts)
            if ranked and ranked[0][1] >= cfg.dedupe_similarity:
                report.skipped_duplicates.append(candidate.candidate_id)
                continue

        replay = build_replay_buffer(episodes, candidate, size=cfg.shadow_eval_size)
        verdict = await run_gate(candidate, replay, evaluator, threshold=cfg.graduation_threshold)
        report.gated[candidate.candidate_id] = verdict

        # Record the verdict on the buffered candidate for audit.
        candidate = record_gate_verdict(store, candidate, verdict)

        if verdict.passed:
            graduate(
                candidate, skills_dir=skills_dir, store=store, manager=manager,
                archive_dir=archive_dir, gate_score=verdict.score,
            )
            report.graduated.append(candidate.candidate_id)
            graduations += 1
            emit(f"graduated '{candidate.skill_name}' (score {verdict.score:.2f})")

    report.cost_usd = meter.spent
    return report


def run_watch_loop(
    tick_fn: Callable[[], TickReport],
    *,
    once: bool = False,
    max_ticks: int | None = None,
    interval_sec: float = 600,
    sleep: Callable[[float], None] | None = None,
    log: _Logger | None = None,
) -> int:
    """Drive `tick_fn` on a schedule. Returns the number of ticks run.

    Stops after one tick if `once`, after `max_ticks` ticks if set, or on
    KeyboardInterrupt. `sleep` is injectable so tests don't actually wait.
    """
    import time as _time

    do_sleep = sleep or _time.sleep
    emit = log or (lambda _m: None)
    n = 0
    try:
        while True:
            tick_fn()
            n += 1
            if once or (max_ticks is not None and n >= max_ticks):
                break
            do_sleep(interval_sec)
    except KeyboardInterrupt:
        emit("watch interrupted; stopping.")
    return n
