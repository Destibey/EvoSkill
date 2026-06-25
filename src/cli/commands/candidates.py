"""evoskill candidates — list and inspect harvested candidate skills."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

console = Console()


@click.command("candidates")
@click.option("--show", "show_id", default=None, help="Print the full SKILL.md for a candidate id.")
@click.option("--status", "status_filter", type=click.Choice(["pending", "graduated", "rejected"]),
              default=None, help="Only show candidates with this status.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Load a specific config TOML file.")
def candidates_cmd(show_id, status_filter, config_path):
    """List buffered candidate skills produced by `evoskill harvest`."""
    from src.cli.config import load_config
    from src.continuous import CandidateStore

    cfg = load_config(config_path=config_path)
    store = CandidateStore(cfg.continuous_candidates_dir)

    if show_id:
        candidate = store.get(show_id)
        if candidate is None:
            console.print(f"[red]Error:[/red] no candidate '{show_id}'.")
            raise SystemExit(1)
        console.print(f"\n  [bold]{candidate.skill_name}[/bold]  ({candidate.candidate_id})")
        console.print(f"  pattern: {candidate.target_pattern}")
        console.print(f"  from {candidate.cluster_size} '{candidate.outcome_focus}' episodes  "
                      f"| status: {candidate.status}\n")
        gate = _gate_label(candidate)
        if gate != "-":
            console.print(f"  gate: {gate}")
            if candidate.extra.get("gate_detail"):
                console.print(f"  [dim]{str(candidate.extra['gate_detail'])[:300]}[/dim]")
            console.print()
        console.print(Syntax(candidate.skill_markdown, "markdown", theme="ansi_dark"))
        return

    candidates = store.list()
    if status_filter:
        candidates = [c for c in candidates if c.status == status_filter]

    if not candidates:
        console.print("  No candidates yet. Run [bold]evoskill harvest[/bold] first.")
        return

    table = Table(box=None, pad_edge=False, show_header=True, header_style="bold")
    table.add_column("Candidate id", min_width=24)
    table.add_column("Skill", min_width=20)
    table.add_column("Size", width=5)
    table.add_column("Status", width=10)
    table.add_column("Gate", width=12)
    table.add_column("Pattern")
    for c in candidates:
        table.add_row(
            c.candidate_id, c.skill_name, str(c.cluster_size), c.status, _gate_label(c),
            (c.target_pattern or "")[:50],
        )
    console.print(table)
    console.print(f"\n  {len(candidates)} candidate(s). "
                  "Inspect one with [bold]evoskill candidates --show <id>[/bold].\n")


def _gate_label(candidate) -> str:
    if "gate_passed" not in candidate.extra:
        return "-"
    status = "pass" if candidate.extra.get("gate_passed") else "fail"
    score = candidate.extra.get("gate_score")
    try:
        return f"{status} {float(score):.2f}"
    except (TypeError, ValueError):
        return status
