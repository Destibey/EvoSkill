"""evoskill gate - evaluate a buffered candidate without graduating it."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command("gate")
@click.argument("candidate_id")
@click.option("--traces", "traces_root", type=click.Path(path_type=Path), default=None,
              help="Trace root to replay (default: configured harbor jobs dir).")
@click.option("--jsonl", "jsonl_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="JSONL trace file to replay. Relative paths are project-relative.")
@click.option("--source", "sources", multiple=True,
              type=click.Choice(["harbor", "goose", "jsonl"]),
              help="Trace source(s) to read. Repeatable. Default: config.")
@click.option("--window", type=int, default=None, help="Max episodes to collect.")
@click.option("--threshold", type=float, default=None, help="Gate pass threshold.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Load a specific config TOML file.")
def gate_cmd(candidate_id, traces_root, jsonl_path, sources, window, threshold, config_path):
    """Run the quality gate and store the verdict, without live skill writeback."""
    from src.agent_profiles import make_surrogate_verifier_options
    from src.cli.config import load_config
    from src.continuous import (
        CandidateStore,
        SurrogateEvaluator,
        TraceCollector,
        build_readers,
        build_replay_buffer,
        record_gate_verdict,
        run_gate,
    )
    from src.harness import Agent, set_sdk
    from src.schemas import SurrogateVerifierResponse

    cfg = load_config(config_path=config_path)
    cont = cfg.continuous
    store = CandidateStore(cfg.continuous_candidates_dir)
    candidate = store.get(candidate_id)
    if candidate is None:
        console.print(f"[red]Error:[/red] no candidate '{candidate_id}'.")
        raise SystemExit(1)

    src_list = list(sources) if sources else cont.trace_sources
    root = str(traces_root) if traces_root else str(cfg.continuous_traces_root)
    jsonl = _resolve_jsonl_path(cfg, jsonl_path)
    window = window if window is not None else cont.harvest_window
    threshold = threshold if threshold is not None else cont.graduation_threshold

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

    episodes = TraceCollector(readers).collect(advance=False, limit=window)
    replay = build_replay_buffer(episodes, candidate, size=cont.shadow_eval_size)

    set_sdk(cfg.harness.name)
    verifier = Agent(
        make_surrogate_verifier_options(
            model=cont.surrogate_model or cfg.harness.model,
            project_root=str(cfg.project_root),
        ),
        SurrogateVerifierResponse,
    )
    verdict = asyncio.run(run_gate(
        candidate,
        replay,
        SurrogateEvaluator(verifier, max_tasks=cont.shadow_eval_size),
        threshold=threshold,
    ))
    record_gate_verdict(
        store,
        candidate,
        verdict,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )

    console.print(
        f"\n  Gate ({verdict.method}): score={verdict.score:.2f} "
        f"threshold={verdict.threshold:.2f} on {verdict.n_tasks} held-out task(s) "
        f"-> {'[green]PASS[/green]' if verdict.passed else '[red]FAIL[/red]'}"
    )
    if verdict.detail:
        console.print(f"  [dim]{verdict.detail[:300]}[/dim]")
    console.print("  Candidate remains buffered; no live skill was written.\n")
    if not verdict.passed:
        raise SystemExit(1)


def _resolve_jsonl_path(cfg, jsonl_path: Path | None) -> Path | None:
    if jsonl_path is None:
        return cfg.continuous_jsonl_path
    path = jsonl_path.expanduser()
    return path if path.is_absolute() else cfg.project_root / path
