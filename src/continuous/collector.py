"""Trace collection for continuous evolution.

This module turns *raw agent traces on disk* into normalized `TaskEpisode`
objects, deduplicated against a persistent watermark so each continuous tick
only processes attempts it hasn't seen before.

Readers (priority order — earlier wins on episode-id collisions):

* `HarborTrajectoryReader` — **primary.** Parses the ATIF (Agent Trajectory
  Interchange Format) `agent/trajectory.json` that every Harbor/Arena harness
  emits (goose, opencode, codex, openhands, ...). Version-tolerant across the
  ATIF schema versions observed in the wild (v1.2 … v1.6+). When a sibling
  `verifier/reward.txt` exists, the reward becomes a ground-truth outcome signal.
* `GooseRawReader` — **fallback.** Reconstructs an episode from the raw
  token-streamed `agent/goose.txt` log, for the rare trial that has no ATIF
  trajectory. Lower fidelity; only used when nothing better exists.
* `JsonlReader` — **generic.** One JSON object per line following a small,
  documented contract, for arbitrary integrations / hooks.

The `TraceCollector` runs a list of readers, dedupes by `episode_id` (within the
batch and against the `TraceCursor`), and optionally advances the watermark.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .episode import ActionStep, Outcome, TaskEpisode, ToolCall
from .signals import no_signal, signal_from_reward

# Placeholder messages harnesses emit when a turn is purely a tool call; never a
# real final answer.
_TOOL_CALL_PLACEHOLDERS = {"[tool call]", "(tool use)", "[tool_call]", "(tool_use)", ""}


class TraceReadError(RuntimeError):
    """Raised when a single trace unit cannot be parsed into an episode."""


# ──────────────────────────────────────────────────────────────────────────────
# ATIF parsing helpers (shared, schema-version-tolerant)
# ──────────────────────────────────────────────────────────────────────────────


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_tool_calls(step: dict[str, Any]) -> list[ToolCall]:
    """Parse a step's `tool_calls`, linking each to its observation content.

    ATIF stores observations separately as `observation.results[]`, linked back
    to a call via `source_call_id` == `tool_call_id`. We fold the matching
    observation content onto each ToolCall so a step is self-contained.
    """
    raw_calls = step.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return []

    # Build call_id -> observation content map.
    obs_by_id: dict[str, str] = {}
    observation = step.get("observation") or {}
    if isinstance(observation, dict):
        for result in observation.get("results") or []:
            if not isinstance(result, dict):
                continue
            call_id = result.get("source_call_id")
            content = result.get("content")
            if call_id is not None and content is not None:
                obs_by_id[str(call_id)] = str(content)

    calls: list[ToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        call_id = raw.get("tool_call_id")
        arguments = raw.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {} if arguments is None else {"value": arguments}
        calls.append(
            ToolCall(
                tool_call_id=str(call_id) if call_id is not None else None,
                function_name=raw.get("function_name"),
                arguments=arguments,
                observation=obs_by_id.get(str(call_id)) if call_id is not None else None,
            )
        )
    return calls


def _parse_step(raw: dict[str, Any]) -> ActionStep:
    return ActionStep(
        step_id=raw.get("step_id"),
        source=str(raw.get("source") or "agent"),
        message=str(raw.get("message") or ""),
        reasoning=str(raw.get("reasoning_content") or ""),
        tool_calls=_parse_tool_calls(raw),
    )


def _derive_task_text(steps: list[dict[str, Any]]) -> str:
    """Best-effort task text from the trace itself.

    ATIF trajectories rarely embed the literal task prompt (it lives in the
    task dataset), but the agent's first step almost always restates the goal in
    its reasoning. We use the first step's reasoning, falling back to its message.
    """
    for step in steps:
        reasoning = str(step.get("reasoning_content") or "").strip()
        if reasoning:
            return reasoning
        message = str(step.get("message") or "").strip()
        if message and message not in _TOOL_CALL_PLACEHOLDERS:
            return message
    return ""


def _derive_final_output(steps: list[dict[str, Any]]) -> str:
    """Best-effort final agent output: the last agent message with real content.

    Note: for tasks graded on a file artifact, the true answer lives in the
    environment, not the transcript — so this is a transcript-level proxy, and
    the verifier reward (not this string) is the outcome of record.
    """
    for step in reversed(steps):
        if str(step.get("source") or "agent") != "agent":
            continue
        message = str(step.get("message") or "").strip()
        if message and message not in _TOOL_CALL_PLACEHOLDERS:
            return message
        # When the last turn is a bare tool call, the answer often sits in its
        # reasoning — prefer that over an earlier step's intro message.
        reasoning = str(step.get("reasoning_content") or "").strip()
        if reasoning:
            return reasoning
    return ""


def parse_atif_trajectory(
    data: dict[str, Any],
    *,
    episode_id: str,
    source: str = "harbor",
    task_id: str | None = None,
    raw_path: str | None = None,
) -> TaskEpisode:
    """Parse one ATIF `trajectory.json` payload into a `TaskEpisode`.

    Tolerant of schema drift: only the common fields (`agent`, `steps`,
    `final_metrics`) are read, and each is defaulted. Outcome is left UNKNOWN —
    the caller attaches a signal (e.g. from a verifier reward) afterwards.
    """
    agent = data.get("agent") or {}
    if not isinstance(agent, dict):
        agent = {}
    steps_raw = data.get("steps") or []
    if not isinstance(steps_raw, list):
        steps_raw = []
    steps_raw = [s for s in steps_raw if isinstance(s, dict)]

    metrics = data.get("final_metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    # Timestamp: present per-step in v1.6, absent in v1.2.
    timestamp = None
    for step in reversed(steps_raw):
        ts = step.get("timestamp")
        if ts:
            timestamp = str(ts)
            break

    return TaskEpisode(
        episode_id=episode_id,
        source=source,
        task_id=task_id,
        task_text=_derive_task_text(steps_raw),
        actions=[_parse_step(s) for s in steps_raw],
        final_output=_derive_final_output(steps_raw),
        outcome=Outcome.UNKNOWN,
        signal=None,
        agent_name=agent.get("name"),
        model_name=agent.get("model_name"),
        prompt_tokens=_coerce_int(metrics.get("total_prompt_tokens")),
        completion_tokens=_coerce_int(metrics.get("total_completion_tokens")),
        cost_usd=_coerce_float(metrics.get("total_cost_usd")),
        num_steps=_coerce_int(metrics.get("total_steps")) or len(steps_raw),
        timestamp=timestamp,
        raw_path=raw_path,
        extra={
            "schema_version": data.get("schema_version"),
            "session_id": data.get("session_id"),
        },
    )


def read_reward_file(trial_dir: Path) -> float | None:
    """Read a Harbor verifier reward from `<trial>/verifier/reward.txt`.

    Returns None when no parseable reward exists (unlabeled trial).
    """
    reward_path = trial_dir / "verifier" / "reward.txt"
    if not reward_path.is_file():
        return None
    try:
        text = reward_path.read_text().strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_result_task_id(trial_dir: Path) -> str | None:
    """Pull the task id from a Harbor `result.json`, if present."""
    result_path = trial_dir / "result.json"
    if not result_path.is_file():
        return None
    try:
        data = json.loads(result_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("task_name") or data.get("source")
    if name:
        return str(name)
    task_id = data.get("task_id")
    if isinstance(task_id, dict) and task_id.get("path"):
        return Path(str(task_id["path"])).name
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Readers
# ──────────────────────────────────────────────────────────────────────────────


class TraceReader(ABC):
    """Base class for trace readers. Each yields normalized `TaskEpisode`s."""

    #: Short source tag stamped onto every episode this reader produces.
    source: str = "trace"

    @abstractmethod
    def read_all(self) -> Iterator[TaskEpisode]:
        """Yield every episode this reader can recover, skipping unparseable units."""
        raise NotImplementedError


class HarborTrajectoryReader(TraceReader):
    """Primary reader: ATIF `agent/trajectory.json` from Harbor/Arena trials.

    Discovers every trial under `jobs_root` by locating `agent/trajectory.json`
    files. For each, parses the ATIF payload, attaches a verifier-reward signal
    when available, and resolves the task id from the sibling `result.json`.
    """

    source = "harbor"

    def __init__(self, jobs_root: str | Path, *, success_threshold: float = 1.0) -> None:
        self.jobs_root = Path(jobs_root).expanduser()
        self.success_threshold = success_threshold

    def _discover_trial_dirs(self) -> list[Path]:
        """Trial dirs are the parents of `agent/trajectory.json`."""
        if not self.jobs_root.is_dir():
            return []
        trials: list[Path] = []
        for traj in self.jobs_root.rglob("trajectory.json"):
            if traj.parent.name == "agent":
                trials.append(traj.parent.parent)
        # Stable order so collection / watermarking is reproducible.
        trials.sort(key=lambda p: str(p))
        return trials

    def parse_trial(self, trial_dir: Path) -> TaskEpisode:
        """Parse a single trial dir into an episode. Raises on unreadable JSON."""
        traj_path = trial_dir / "agent" / "trajectory.json"
        try:
            data = json.loads(traj_path.read_text())
        except (OSError, ValueError) as exc:
            raise TraceReadError(f"unreadable trajectory {traj_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise TraceReadError(f"trajectory is not a JSON object: {traj_path}")

        episode_id = trial_dir.name or data.get("session_id") or _hash_path(traj_path)
        episode = parse_atif_trajectory(
            data,
            episode_id=str(episode_id),
            source=self.source,
            task_id=_read_result_task_id(trial_dir),
            raw_path=str(traj_path),
        )

        reward = read_reward_file(trial_dir)
        if reward is not None:
            episode.signal = signal_from_reward(reward, success_threshold=self.success_threshold)
        else:
            episode.signal = no_signal()
        episode.outcome = episode.signal.outcome
        return episode

    def read_all(self) -> Iterator[TaskEpisode]:
        for trial_dir in self._discover_trial_dirs():
            try:
                yield self.parse_trial(trial_dir)
            except TraceReadError:
                # A single corrupt trial must not abort the whole collection.
                continue


class GooseRawReader(TraceReader):
    """Fallback reader: reconstruct an episode from raw `agent/goose.txt`.

    `goose.txt` is a token-streamed JSONL log: each line carries a fragment of a
    message (`thinking`/`text` deltas, tool requests). We group fragments by
    message id and concatenate them into coarse `ActionStep`s. This is lossy and
    only meant for trials lacking an ATIF trajectory; `HarborTrajectoryReader`
    takes precedence when both exist (same `episode_id` ⇒ deduped).
    """

    source = "goose"

    def __init__(self, jobs_root: str | Path, *, success_threshold: float = 1.0) -> None:
        self.jobs_root = Path(jobs_root).expanduser()
        self.success_threshold = success_threshold

    def _discover_logs(self) -> list[Path]:
        if not self.jobs_root.is_dir():
            return []
        logs = [p for p in self.jobs_root.rglob("goose.txt") if p.parent.name == "agent"]
        logs.sort(key=lambda p: str(p))
        return logs

    def parse_log(self, log_path: Path) -> TaskEpisode:
        trial_dir = log_path.parent.parent
        try:
            lines = log_path.read_text().splitlines()
        except OSError as exc:
            raise TraceReadError(f"unreadable goose log {log_path}: {exc}") from exc

        # message id -> {"thinking": [...], "text": [...], "tools": [...]}
        buffers: dict[str, dict[str, list]] = {}
        order: list[str] = []
        model_name: str | None = None

        for line in lines:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get("type") != "message":
                continue
            message = event.get("message") or {}
            if not isinstance(message, dict):
                continue
            mid = str(message.get("id") or "")
            if not mid:
                continue
            if mid not in buffers:
                buffers[mid] = {"thinking": [], "text": [], "tools": []}
                order.append(mid)
            for chunk in message.get("content") or []:
                if not isinstance(chunk, dict):
                    continue
                ctype = chunk.get("type")
                if ctype == "thinking":
                    buffers[mid]["thinking"].append(str(chunk.get("thinking") or ""))
                elif ctype == "text":
                    buffers[mid]["text"].append(str(chunk.get("text") or ""))
                elif ctype in ("toolRequest", "tool_use", "toolUse"):
                    buffers[mid]["tools"].append(chunk)

        steps: list[ActionStep] = []
        for i, mid in enumerate(order, start=1):
            buf = buffers[mid]
            tool_calls = [
                ToolCall(
                    tool_call_id=str(t.get("id")) if t.get("id") is not None else None,
                    function_name=(t.get("toolCall") or {}).get("name")
                    if isinstance(t.get("toolCall"), dict)
                    else t.get("name"),
                )
                for t in buf["tools"]
                if isinstance(t, dict)
            ]
            steps.append(
                ActionStep(
                    step_id=i,
                    source="agent",
                    message="".join(buf["text"]).strip(),
                    reasoning="".join(buf["thinking"]).strip(),
                    tool_calls=tool_calls,
                )
            )

        if not steps:
            raise TraceReadError(f"no reconstructable messages in {log_path}")

        episode = TaskEpisode(
            episode_id=trial_dir.name or _hash_path(log_path),
            source=self.source,
            task_id=_read_result_task_id(trial_dir),
            task_text=steps[0].reasoning or steps[0].message,
            actions=steps,
            final_output=next(
                (s.message for s in reversed(steps) if s.message), ""
            ),
            agent_name="goose",
            model_name=model_name,
            num_steps=len(steps),
            raw_path=str(log_path),
        )

        reward = read_reward_file(trial_dir)
        if reward is not None:
            episode.signal = signal_from_reward(reward, success_threshold=self.success_threshold)
        else:
            episode.signal = no_signal()
        episode.outcome = episode.signal.outcome
        return episode

    def read_all(self) -> Iterator[TaskEpisode]:
        for log_path in self._discover_logs():
            try:
                yield self.parse_log(log_path)
            except TraceReadError:
                continue


class JsonlReader(TraceReader):
    """Generic reader: one JSON object per line.

    Contract (all fields optional except `task`):

        {
          "episode_id": "...",          # stable id; derived from content if omitted
          "task_id": "...",
          "task": "the instruction",    # required
          "output": "final answer",
          "reward": 1.0,                # → ground-truth verifier-style signal
          "outcome": "success",         # explicit override if no reward
          "skills_active": ["s1"],
          "agent": "claude-code",
          "model": "...",
          "steps": [{"source": "agent", "message": "...", "reasoning": "..."}]
        }

    Lines that are blank, non-JSON, or missing `task` are skipped.
    """

    source = "jsonl"

    def __init__(
        self, path: str | Path, *, source: str | None = None, success_threshold: float = 1.0
    ) -> None:
        self.path = Path(path).expanduser()
        if source:
            self.source = source
        self.success_threshold = success_threshold

    def parse_record(self, record: dict[str, Any], *, index: int) -> TaskEpisode:
        task_text = str(record.get("task") or "").strip()
        if not task_text:
            raise TraceReadError(f"record {index} missing 'task'")

        steps = []
        for j, raw in enumerate(record.get("steps") or [], start=1):
            if not isinstance(raw, dict):
                continue
            steps.append(
                ActionStep(
                    step_id=raw.get("step_id") or j,
                    source=str(raw.get("source") or "agent"),
                    message=str(raw.get("message") or ""),
                    reasoning=str(raw.get("reasoning") or raw.get("reasoning_content") or ""),
                )
            )

        episode_id = str(
            record.get("episode_id")
            or _hash_text(f"{self.path}:{index}:{task_text}")
        )

        episode = TaskEpisode(
            episode_id=episode_id,
            source=self.source,
            task_id=record.get("task_id"),
            task_text=task_text,
            actions=steps,
            final_output=str(record.get("output") or ""),
            skills_active=[str(s) for s in (record.get("skills_active") or [])],
            agent_name=record.get("agent"),
            model_name=record.get("model"),
            prompt_tokens=_coerce_int(record.get("prompt_tokens")),
            completion_tokens=_coerce_int(record.get("completion_tokens")),
            cost_usd=_coerce_float(record.get("cost_usd")),
            num_steps=len(steps),
            timestamp=str(record["timestamp"]) if record.get("timestamp") else None,
            raw_path=str(self.path),
            extra=record.get("extra") if isinstance(record.get("extra"), dict) else {},
        )

        reward = record.get("reward")
        if reward is not None:
            episode.signal = signal_from_reward(
                _coerce_float(reward), success_threshold=self.success_threshold
            )
        elif record.get("outcome"):
            outcome = _outcome_from_str(str(record["outcome"]))
            episode.signal = no_signal(evidence=f"explicit outcome '{record['outcome']}'")
            episode.signal.outcome = outcome
            episode.signal.confidence = 0.5
        else:
            episode.signal = no_signal()
        episode.outcome = episode.signal.outcome
        return episode

    def read_all(self) -> Iterator[TaskEpisode]:
        if not self.path.is_file():
            return
        with self.path.open() as fh:
            for index, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                try:
                    yield self.parse_record(record, index=index)
                except TraceReadError:
                    continue


def _outcome_from_str(value: str) -> Outcome:
    v = value.strip().lower()
    if v in ("success", "pass", "passed", "true", "1"):
        return Outcome.SUCCESS
    if v in ("failure", "fail", "failed", "false", "0"):
        return Outcome.FAILURE
    return Outcome.UNKNOWN


def _hash_path(path: Path) -> str:
    return _hash_text(str(path))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────────
# Watermark cursor + collector
# ──────────────────────────────────────────────────────────────────────────────


class TraceCursor:
    """Persistent watermark of which episodes have already been collected.

    Stored as JSON at `path` (default `.evoskill/continuous/cursor.json`). Keeps
    a set of seen `episode_id`s so each tick only processes new attempts.

    For the corpus sizes continuous evolution targets (hundreds–thousands of
    episodes) a flat id set is fine; this can later be compacted to a per-source
    timestamp watermark if the file grows large.
    """

    DEFAULT_PATH = Path(".evoskill") / "continuous" / "cursor.json"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else self.DEFAULT_PATH
        self._seen: set[str] = set()
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            self._seen = set()
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self._seen = set()
            return
        seen = data.get("seen") if isinstance(data, dict) else None
        self._seen = {str(x) for x in seen} if isinstance(seen, list) else set()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"seen": sorted(self._seen), "count": len(self._seen)}
        self.path.write_text(json.dumps(payload, indent=2))

    def seen(self, episode_id: str) -> bool:
        return episode_id in self._seen

    def mark(self, episode_id: str) -> None:
        self._seen.add(episode_id)

    def __len__(self) -> int:
        return len(self._seen)


class TraceCollector:
    """Run readers, normalize, dedup by `episode_id`, advance the watermark.

    Reader order is priority: if two readers produce the same `episode_id`
    (e.g. a trial with both an ATIF trajectory and a raw goose log), the earlier
    reader wins and the later duplicate is dropped.
    """

    def __init__(
        self,
        readers: list[TraceReader],
        *,
        cursor: TraceCursor | None = None,
    ) -> None:
        self.readers = readers
        self.cursor = cursor

    def collect(self, *, advance: bool = True, limit: int | None = None) -> list[TaskEpisode]:
        """Collect new episodes across all readers.

        Args:
            advance: mark collected episodes in the cursor and persist it.
            limit: stop after this many *new* episodes (None = unlimited).

        Returns:
            New, deduplicated episodes in reader-priority then discovery order.
        """
        seen_in_batch: set[str] = set()
        episodes: list[TaskEpisode] = []

        for reader in self.readers:
            for episode in reader.read_all():
                eid = episode.episode_id
                if eid in seen_in_batch:
                    continue
                if self.cursor is not None and self.cursor.seen(eid):
                    continue
                seen_in_batch.add(eid)
                episodes.append(episode)
                if limit is not None and len(episodes) >= limit:
                    break
            if limit is not None and len(episodes) >= limit:
                break

        if advance and self.cursor is not None:
            for episode in episodes:
                self.cursor.mark(episode.episode_id)
            self.cursor.save()

        return episodes
