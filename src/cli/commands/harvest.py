"""evoskill harvest — distill candidate skills from real usage traces.

Offline Phase-1 entry point for continuous evolution: collect a window of agent
trajectories, cluster recurring failures (or successes), and distill one
reusable candidate skill per cluster into the review buffer. Candidates are NOT
added to the live skill library — review them with `evoskill candidates`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()

_FOCUS_TO_OUTCOME = {"failure": "FAILURE", "success": "SUCCESS"}


@click.command("harvest")
@click.option("--traces", "traces_root", type=click.Path(path_type=Path), default=None,
              help="Trace root to harvest (default: configured harbor jobs dir).")
@click.option("--jsonl", "jsonl_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="JSONL trace file to harvest. Relative paths are project-relative.")
@click.option("--source", "sources", multiple=True,
              type=click.Choice(["harbor", "goose", "jsonl"]),
              help="Trace source(s) to read. Repeatable. Default: config.")
@click.option("--window", type=int, default=None, help="Max episodes to collect.")
@click.option("--min-cluster-size", type=int, default=None,
              help="Minimum episodes per cluster to distill a candidate.")
@click.option("--max-candidates", type=int, default=None,
              help="Cap the number of candidates distilled (largest clusters first).")
@click.option("--focus", type=click.Choice(["failure", "success", "both"]), default=None,
              help="Mine failures (capability gaps), successes, or both.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Cluster only — no LLM distillation, no candidates written.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Load a specific config TOML file.")
def harvest_cmd(traces_root, jsonl_path, sources, window, min_cluster_size, max_candidates,
                focus, dry_run, config_path):
    """Distill candidate skills from a window of usage traces."""
    from src.cli.config import load_config
    from src.continuous import (
        CandidateStore,
        Outcome,
        TraceCollector,
        build_readers,
        cluster_episodes,
        harvest as run_harvest,
    )

    cfg = load_config(config_path=config_path)
    cont = cfg.continuous

    # Resolve effective parameters (CLI overrides config).
    src_list = list(sources) if sources else cont.trace_sources
    root = str(traces_root) if traces_root else str(cfg.continuous_traces_root)
    jsonl = _resolve_jsonl_path(cfg, jsonl_path)
    window = window if window is not None else cont.harvest_window
    min_cluster_size = min_cluster_size if min_cluster_size is not None else cont.min_cluster_size
    max_candidates = max_candidates if max_candidates is not None else cont.max_candidates
    focus = focus or cont.focus

    readers = build_readers(
        src_list,
        traces_root=root,
        jsonl_path=str(jsonl) if jsonl else None,
        success_threshold=cont.success_threshold,
    )
    if not readers:
        console.print(
            f"[red]Error:[/red] no usable trace sources for {src_list} at [bold]{root}[/bold].\n"
            "  Point --traces at a Harbor jobs dir, pass --jsonl, or set continuous.jsonl_path."
        )
        raise SystemExit(1)

    focuses = [Outcome.FAILURE, Outcome.SUCCESS] if focus == "both" else [
        getattr(Outcome, _FOCUS_TO_OUTCOME[focus])
    ]

    jsonl_line = f"\n  jsonl: {jsonl}" if jsonl else ""
    console.print(
        f"\n  [bold]EvoSkill harvest[/bold] — sources={src_list}  focus={focus}  "
        f"window={window}  min_cluster={min_cluster_size}\n  traces: {root}{jsonl_line}\n"
    )

    # ── dry run: collect + cluster only ──
    if dry_run:
        episodes = TraceCollector(readers).collect(advance=False, limit=window)
        console.print(f"  Collected {len(episodes)} episode(s).")
        for foc in focuses:
            clusters = cluster_episodes(
                episodes, focus=foc, min_cluster_size=min_cluster_size,
                similarity_threshold=cont.similarity_threshold, max_clusters=max_candidates,
            )
            _print_clusters(foc.value, clusters)
        console.print("\n  [dim]dry-run: no candidates written[/dim]\n")
        return

    # ── full harvest: distill + write candidates ──
    from src.harness import Agent, set_sdk
    from src.agent_profiles import make_skill_distiller_options
    from src.schemas import SkillDistillerResponse

    set_sdk(cfg.harness.name)
    distiller = Agent(
        make_skill_distiller_options(
            model=cont.distiller_model or cfg.harness.model,
            project_root=str(cfg.project_root),
        ),
        SkillDistillerResponse,
    )
    store = CandidateStore(cfg.continuous_candidates_dir)

    total_candidates = 0
    for foc in focuses:
        result = asyncio.run(
            run_harvest(
                readers=readers,
                distiller=distiller,
                store=store,
                window=window,
                min_cluster_size=min_cluster_size,
                similarity_threshold=cont.similarity_threshold,
                focus=foc,
                max_candidates=max_candidates,
                concurrency=cont.concurrency,
                advance_cursor=False,
                log=lambda msg: console.print(f"  {msg}"),
            )
        )
        total_candidates += result.num_candidates

    console.print(
        f"\n  Wrote [bold]{total_candidates}[/bold] candidate(s) to "
        f"{cfg.continuous_candidates_dir}\n"
        "  Review with [bold]evoskill candidates[/bold].\n"
    )


def _print_clusters(focus: str, clusters) -> None:
    if not clusters:
        console.print(f"  No '{focus}' clusters met the threshold.")
        return
    table = Table(box=None, pad_edge=False, show_header=True, header_style="bold",
                  title=f"{focus} clusters")
    table.add_column("Size", width=6)
    table.add_column("Top terms")
    table.add_column("Example task")
    for c in clusters:
        example = c.episodes[0].task_text[:60].replace("\n", " ") if c.episodes else ""
        table.add_row(str(c.size), ", ".join(c.top_terms[:5]), example)
    console.print(table)


def _resolve_jsonl_path(cfg, jsonl_path: Path | None) -> Path | None:
    if jsonl_path is None:
        return cfg.continuous_jsonl_path
    path = jsonl_path.expanduser()
    return path if path.is_absolute() else cfg.project_root / path
