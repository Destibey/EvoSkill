"""evoskill watch — the always-on continuous-evolution daemon.

Repeats one `run_tick` (collect → credit → cluster → distill → [auto: gate →
graduate] → deprecation → watermark) on a schedule. `review` mode (default) only
discovers candidates; `auto` mode gates and graduates them with safety rails
(dedup-guard, rate limit, cost ceiling).

    evoskill watch              # loop forever (Ctrl-C to stop)
    evoskill watch --once       # a single tick (good for cron / testing)
    evoskill watch --max-ticks 5
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console

console = Console()

_FOCUS = {"failure": "FAILURE", "success": "SUCCESS", "both": "FAILURE"}


@click.command("watch")
@click.option("--once", is_flag=True, default=False, help="Run a single tick and exit.")
@click.option("--max-ticks", type=int, default=None, help="Stop after this many ticks.")
@click.option("--interval", "interval_sec", type=int, default=None,
              help="Seconds between ticks (overrides config poll_interval_sec).")
@click.option("--mode", type=click.Choice(["review", "auto"]), default=None,
              help="Override graduation mode for this run.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Load a specific config TOML file.")
def watch_cmd(once, max_ticks, interval_sec, mode, config_path):
    """Run the continuous-evolution daemon."""
    from src.cli.config import load_config
    from src.continuous import (
        CandidateStore,
        Outcome,
        SkillStatsStore,
        TickConfig,
        TraceCursor,
        build_readers,
        make_similarity_backend,
        run_tick,
        run_watch_loop,
    )
    from src.harness import Agent, set_sdk

    cfg = load_config(config_path=config_path)
    cont = cfg.continuous
    mode = mode or cont.graduation_mode
    set_sdk(cfg.harness.name)

    readers = build_readers(
        cont.trace_sources,
        traces_root=str(cfg.continuous_traces_root),
        jsonl_path=str(cfg.continuous_jsonl_path) if cfg.continuous_jsonl_path else None,
        success_threshold=cont.success_threshold,
    )
    if not readers:
        console.print(
            f"[red]Error:[/red] no usable trace sources for {cont.trace_sources} "
            f"at {cfg.continuous_traces_root}."
        )
        raise SystemExit(1)

    from src.agent_profiles import make_skill_distiller_options
    from src.schemas import SkillDistillerResponse

    distiller = Agent(
        make_skill_distiller_options(
            model=cont.distiller_model or cfg.harness.model, project_root=str(cfg.project_root)),
        SkillDistillerResponse,
    )

    cursor = TraceCursor(cfg.evoskill_dir / "continuous" / "cursor.json")
    store = CandidateStore(cfg.continuous_candidates_dir)
    stats_store = SkillStatsStore(cfg.continuous_skill_stats_path)

    tick_config = TickConfig(
        mode=mode,
        window=cont.harvest_window,
        focus=getattr(Outcome, _FOCUS[cont.focus]),
        min_cluster_size=cont.min_cluster_size,
        similarity_threshold=cont.similarity_threshold,
        max_candidates=cont.max_candidates,
        concurrency=cont.concurrency,
        graduation_threshold=cont.graduation_threshold,
        shadow_eval_size=cont.shadow_eval_size,
        max_graduations=cont.max_graduations_per_window,
        dedupe_similarity=cont.lifecycle.dedupe_similarity,
        deprecation_baseline=cont.lifecycle.deprecation_baseline,
        deprecation_strikes=cont.lifecycle.deprecation_strikes,
        auto_deprecate=cont.auto_deprecate,
        cost_ceiling=cont.cost_ceiling_usd_per_tick or None,
    )

    # Gate/graduation collaborators are only needed in auto mode.
    verifier = None
    manager = None
    similarity_backend = None
    if mode == "auto":
        from src.agent_profiles import make_surrogate_verifier_options
        from src.harness.provider_auth import resolve_provider_api_key
        from src.registry import ProgramManager
        from src.schemas import SurrogateVerifierResponse

        verifier = Agent(
            make_surrogate_verifier_options(
                model=cont.surrogate_model or cfg.harness.model, project_root=str(cfg.project_root)),
            SurrogateVerifierResponse,
        )
        manager = ProgramManager(cwd=cfg.project_root)
        api_key, _ = resolve_provider_api_key(cont.lifecycle.embedding_provider)
        similarity_backend = make_similarity_backend(
            backend=cont.lifecycle.similarity_backend,
            provider=cont.lifecycle.embedding_provider,
            model=cont.lifecycle.embedding_model,
            api_key=api_key,
            cache_path=str(cfg.continuous_embeddings_cache_path),
        )

    interval = interval_sec if interval_sec is not None else cont.poll_interval_sec
    console.print(
        f"\n  [bold]EvoSkill watch[/bold] — mode={mode}  interval={interval}s  "
        f"traces={cfg.continuous_traces_root}\n"
    )

    def _one_tick():
        report = asyncio.run(run_tick(
            readers=readers, store=store, distiller=distiller, cursor=cursor,
            verifier=verifier, skills_dir=cfg.skills_dir, stats_store=stats_store,
            similarity_backend=similarity_backend, manager=manager,
            archive_dir=cfg.continuous_deprecated_dir, config=tick_config,
            log=lambda m: console.print(f"  {m}"),
        ))
        summary = (f"tick: {report.episodes_collected} episodes, "
                   f"{report.num_candidates} candidate(s)")
        if mode == "auto":
            summary += f", {report.num_graduated} graduated"
            if report.skipped_duplicates:
                summary += f", {len(report.skipped_duplicates)} dup-skipped"
            if report.stopped_reason:
                summary += f" [stopped: {report.stopped_reason}]"
        console.print(f"  [green]{summary}[/green]  (${report.cost_usd:.4f})")
        return report

    ticks = run_watch_loop(
        _one_tick, once=once, max_ticks=max_ticks, interval_sec=interval,
        log=lambda m: console.print(f"  {m}"),
    )
    console.print(f"\n  Ran {ticks} tick(s).\n")
