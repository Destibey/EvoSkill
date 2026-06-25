"""The quality gate: judge a candidate before it can graduate.

The gate enforces the one invariant continuous evolution cannot live without:
**a candidate never goes live on the strength of the episodes that created it.**
It is judged on a held-out *replay buffer* — episodes the candidate was not
distilled from — exactly as the batch loop scores on a validation set it never
proposed against.

Phase 3 ships the **surrogate verifier** evaluator (research direction A): an
isolated verifier agent synthesizes assertions from the candidate skill and the
held-out task descriptions and decides whether the skill is correct and
generalizable — no ground truth, no re-running the agent. The `GateEvaluator`
interface lets a shadow-re-eval evaluator drop in later without touching the
policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .candidates import Candidate, CandidateStore
from .episode import TaskEpisode
from .signals import clamp01


@dataclass
class GateTask:
    """A held-out task shown to the evaluator — description only, no answer."""

    task_text: str
    task_id: str | None = None


@dataclass
class EvalOutcome:
    """An evaluator's judgement of a candidate."""

    score: float
    verdict: bool
    assertions: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class GateVerdict:
    """The gate's decision for one candidate."""

    passed: bool
    method: str
    score: float
    threshold: float
    n_tasks: int
    assertions: list[str] = field(default_factory=list)
    detail: str = ""


def build_replay_buffer(
    episodes: list[TaskEpisode],
    candidate: Candidate,
    *,
    size: int = 10,
) -> list[TaskEpisode]:
    """Held-out episodes for judging `candidate`.

    Excludes every episode the candidate was distilled from (`candidate.episode_ids`)
    so the gate measures generalization, not recall. Deterministic: returns the
    first `size` remaining episodes in their given order.
    """
    excluded = set(candidate.episode_ids)
    held_out = [e for e in episodes if e.episode_id not in excluded]
    return held_out[:size]


class GateEvaluator(Protocol):
    """Judges a candidate against held-out tasks. `method` names the strategy."""

    method: str

    async def evaluate(self, candidate: Candidate, tasks: list[GateTask]) -> EvalOutcome:  # pragma: no cover - protocol
        ...


def build_surrogate_query(
    candidate: Candidate,
    tasks: list[GateTask],
    *,
    max_tasks: int = 10,
    task_chars: int = 400,
) -> str:
    """Render the candidate + held-out tasks into a prompt for the verifier."""
    lines = [
        "## Candidate skill (SKILL.md)",
        candidate.skill_markdown.strip(),
        "",
        "## Held-out task descriptions (the skill was NOT distilled from these)",
    ]
    shown = tasks[:max_tasks]
    if not shown:
        lines.append("(none available — no independent replay evidence)")
    for i, t in enumerate(shown, start=1):
        text = t.task_text.strip()
        if len(text) > task_chars:
            text = text[: task_chars - 1].rstrip() + "…"
        lines.append(f"{i}. {text or '(no description)'}")
    if len(tasks) > max_tasks:
        lines.append(f"(+{len(tasks) - max_tasks} more not shown)")
    lines += [
        "",
        "Synthesize assertions, then decide whether this skill is correct and "
        "generalizable (not memorized). Return score, verdict, assertions, reasoning.",
    ]
    return "\n".join(lines)


class SurrogateEvaluator:
    """Judge a candidate with an isolated surrogate-verifier agent."""

    method = "surrogate"

    def __init__(self, verifier: Any, *, max_tasks: int = 10) -> None:
        self._verifier = verifier
        self.max_tasks = max_tasks

    async def evaluate(self, candidate: Candidate, tasks: list[GateTask]) -> EvalOutcome:
        query = build_surrogate_query(candidate, tasks, max_tasks=self.max_tasks)
        try:
            trace = await self._verifier.run(query)
        except Exception as exc:  # noqa: BLE001 - a verifier failure must not crash the gate
            return EvalOutcome(score=0.0, verdict=False, detail=f"verifier error: {exc}")
        output = getattr(trace, "output", None)
        if output is None:
            return EvalOutcome(score=0.0, verdict=False, detail="verifier produced no output")
        return EvalOutcome(
            score=clamp01(float(getattr(output, "score", 0.0) or 0.0)),
            verdict=bool(getattr(output, "verdict", False)),
            assertions=list(getattr(output, "assertions", []) or []),
            detail=str(getattr(output, "reasoning", "") or ""),
        )


async def run_gate(
    candidate: Candidate,
    replay_episodes: list[TaskEpisode],
    evaluator: GateEvaluator,
    *,
    threshold: float = 0.6,
) -> GateVerdict:
    """Run the gate: evaluate the candidate on the held-out buffer, apply the policy.

    A candidate passes only if the evaluator returns `verdict=True` AND its score
    meets `threshold`. Both conditions guard against weak passes.
    """
    tasks = [GateTask(task_text=e.task_text, task_id=e.task_id) for e in replay_episodes]
    if not tasks:
        return GateVerdict(
            passed=False,
            method=evaluator.method,
            score=0.0,
            threshold=threshold,
            n_tasks=0,
            detail="no held-out replay tasks; cannot prove generalization",
        )
    outcome = await evaluator.evaluate(candidate, tasks)
    passed = bool(outcome.verdict) and outcome.score >= threshold
    return GateVerdict(
        passed=passed,
        method=evaluator.method,
        score=outcome.score,
        threshold=threshold,
        n_tasks=len(tasks),
        assertions=outcome.assertions,
        detail=outcome.detail,
    )


def record_gate_verdict(
    store: CandidateStore,
    candidate: Candidate,
    verdict: GateVerdict,
    *,
    evaluated_at: str | None = None,
) -> Candidate:
    """Persist a non-live gate verdict on a buffered candidate for audit/review."""
    extra = dict(candidate.extra)
    extra.update(
        {
            "gate_method": verdict.method,
            "gate_score": verdict.score,
            "gate_threshold": verdict.threshold,
            "gate_passed": verdict.passed,
            "gate_n_tasks": verdict.n_tasks,
            "gate_assertions": verdict.assertions,
            "gate_detail": verdict.detail,
        }
    )
    if evaluated_at is not None:
        extra["gate_evaluated_at"] = evaluated_at
    updated = candidate.model_copy(update={"extra": extra})
    store.save(updated)
    return updated
