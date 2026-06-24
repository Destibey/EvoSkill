"""evoskill complaint — record operator dissatisfaction as a JSONL trace.

This is the smallest bridge from "the operator is unhappy with a skill" to the
continuous-evolution pipeline. It writes a local JSONL failure episode that can
be mined by `evoskill harvest --source jsonl --dry-run` without calling an LLM
or touching live skill folders.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from rich.console import Console

console = Console()
_ID_SAFE = re.compile(r"[^0-9A-Za-z]+")


@click.command("complaint")
@click.argument("complaint_parts", nargs=-1)
@click.option("--skill", "skills", multiple=True,
              help="Skill name believed to be involved. Repeatable.")
@click.option("--task", "task_context", default="",
              help="Optional task/user context that triggered the complaint.")
@click.option("--output", "output_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="JSONL output path. Relative paths are project-relative.")
@click.option("--episode-id", default=None,
              help="Stable id to store in the JSONL record. Generated if omitted.")
@click.option("--timestamp", default=None,
              help="Timestamp for the record, e.g. 2026-06-24T00:00:00Z.")
@click.option("--agent", default="operator-complaint",
              help="Agent/provenance name stamped onto the record.")
@click.option("--model", "model_name", default=None,
              help="Optional model name, if the complaint refers to a specific run.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Load a specific config TOML file.")
def complaint_cmd(
    complaint_parts,
    skills,
    task_context,
    output_path,
    episode_id,
    timestamp,
    agent,
    model_name,
    config_path,
):
    """Append one operator complaint as a failure episode JSONL record."""
    from src.cli.config import load_config

    complaint = " ".join(complaint_parts).strip()
    if not complaint:
        console.print("[red]Error:[/red] complaint text is required.")
        raise SystemExit(2)

    cfg = load_config(config_path=config_path)
    out = _resolve_output_path(cfg, output_path)
    skill_list = [s.strip() for s in skills if s.strip()]
    record = build_complaint_record(
        complaint=complaint,
        skills=skill_list,
        task_context=task_context.strip(),
        episode_id=episode_id,
        timestamp=timestamp,
        agent=agent,
        model_name=model_name,
    )
    append_jsonl_record(out, record)

    console.print(f"\n  [green]Recorded complaint trace[/green] → {out}")
    console.print(f"  episode_id={record['episode_id']}  outcome=failure  skills={skill_list or '-'}")
    if not cfg.continuous_jsonl_path:
        console.print(
            "  To harvest this source, set continuous.trace_sources = [\"jsonl\"] "
            "and jsonl_path = \".evoskill/continuous/complaints.jsonl\"."
        )
    console.print()


def build_complaint_record(
    *,
    complaint: str,
    skills: list[str],
    task_context: str = "",
    episode_id: str | None = None,
    timestamp: str | None = None,
    agent: str = "operator-complaint",
    model_name: str | None = None,
) -> dict[str, Any]:
    """Build a JsonlReader-compatible failure record for one complaint."""
    ts = timestamp or _utc_timestamp()
    eid = episode_id or _episode_id(complaint=complaint, skills=skills,
                                   task_context=task_context, timestamp=ts)
    task_lines = [f"Operator complaint: {complaint}"]
    if task_context:
        task_lines.append(f"Task context: {task_context}")
    if skills:
        task_lines.append(f"Affected skills: {', '.join(skills)}")

    return {
        "episode_id": eid,
        "task_id": "operator-complaint",
        "task": "\n".join(task_lines),
        "output": complaint,
        "outcome": "failure",
        "skills_active": skills,
        "agent": agent,
        "model": model_name,
        "timestamp": ts,
        "steps": [
            {"source": "user", "message": complaint},
            {
                "source": "environment",
                "message": "Recorded operator dissatisfaction as a continuous-evolution failure trace.",
            },
        ],
        "extra": {
            "kind": "operator_complaint",
            "complaint": complaint,
            "task_context": task_context,
        },
    }


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _resolve_output_path(cfg, output_path: Path | None) -> Path:
    if output_path is None:
        return cfg.continuous_complaints_path
    path = output_path.expanduser()
    return path if path.is_absolute() else cfg.project_root / path


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _episode_id(*, complaint: str, skills: list[str], task_context: str, timestamp: str) -> str:
    payload = json.dumps(
        {
            "complaint": complaint,
            "skills": skills,
            "task_context": task_context,
            "timestamp": timestamp,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    ts = _ID_SAFE.sub("", timestamp)[:16] or "unknown"
    return f"complaint-{ts}-{digest}"
