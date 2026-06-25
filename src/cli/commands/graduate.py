"""evoskill graduate / reject — promote (or discard) a candidate skill.

`evoskill graduate <id>` runs the quality gate (surrogate verifier on a held-out
replay buffer) and, on pass, installs the skill into the live library and
snapshots a `program/*` branch. `--force` skips the gate. `--no-branch` is
refused for live graduation because it would leave no rollback snapshot.

`evoskill reject <id>` marks a candidate rejected without touching the library.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command("graduate")
@click.argument("candidate_id")
@click.option("--force", is_flag=True, default=False, help="Skip the gate and graduate anyway.")
@click.option("--no-branch", is_flag=True, default=False,
              help="Deprecated unsafe mode; live graduation now requires a program/* branch.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Load a specific config TOML file.")
def graduate_cmd(candidate_id, force, no_branch, config_path):
    """Run the gate and graduate a candidate into the live skill library."""
    from src.cli.config import load_config
    from src.continuous import (
        CandidateStore,
        SurrogateEvaluator,
        TraceCollector,
        baseline_candidate_from_library,
        build_readers,
        build_replay_buffer,
        graduate,
        record_gate_verdict,
        run_gate,
    )

    cfg = load_config(config_path=config_path)
    store = CandidateStore(cfg.continuous_candidates_dir)
    candidate = store.get(candidate_id)
    if candidate is None:
        console.print(f"[red]Error:[/red] no candidate '{candidate_id}'.")
        raise SystemExit(1)

    if no_branch:
        console.print(
            "[red]Error:[/red] refusing to graduate without a program/* snapshot.\n"
            "  Continuous graduation writes live skills, so rollback evidence is required."
        )
        raise SystemExit(1)

    gate_score = None
    if not force:
        from src.harness import Agent, set_sdk
        from src.agent_profiles import make_surrogate_verifier_options
        from src.schemas import SurrogateVerifierResponse

        set_sdk(cfg.harness.name)
        readers = build_readers(
            cfg.continuous.trace_sources,
            traces_root=str(cfg.continuous_traces_root),
            jsonl_path=str(cfg.continuous_jsonl_path) if cfg.continuous_jsonl_path else None,
            success_threshold=cfg.continuous.success_threshold,
        )
        episodes = TraceCollector(readers).collect(advance=False, limit=cfg.continuous.harvest_window)
        replay = build_replay_buffer(episodes, candidate, size=cfg.continuous.shadow_eval_size)
        baseline = baseline_candidate_from_library(candidate, cfg.skills_dir)

        verifier = Agent(
            make_surrogate_verifier_options(
                model=cfg.continuous.surrogate_model or cfg.harness.model,
                project_root=str(cfg.project_root),
            ),
            SurrogateVerifierResponse,
        )
        verdict = asyncio.run(run_gate(
            candidate, replay, SurrogateEvaluator(verifier),
            threshold=cfg.continuous.graduation_threshold, baseline=baseline,
        ))
        candidate = record_gate_verdict(store, candidate, verdict)
        gate_score = verdict.score
        console.print(
            f"\n  Gate ({verdict.method}): score={verdict.score:.2f} "
            f"threshold={verdict.threshold:.2f} on {verdict.n_tasks} held-out task(s) "
            f"→ {'[green]PASS[/green]' if verdict.passed else '[red]FAIL[/red]'}"
        )
        if verdict.detail:
            console.print(f"  [dim]{verdict.detail[:300]}[/dim]")
        if verdict.baseline_score is not None:
            console.print(
                f"  Baseline {verdict.baseline_name}: score={verdict.baseline_score:.2f} "
                f"improvement={verdict.improvement:+.2f}"
            )
        if not verdict.passed:
            console.print("\n  Not graduated. Re-run with --force to override.\n")
            raise SystemExit(1)

    from src.registry import ProgramManager
    manager = ProgramManager(cwd=cfg.project_root)

    result = graduate(
        candidate,
        skills_dir=cfg.skills_dir,
        store=store,
        manager=manager,
        archive_dir=cfg.continuous_deprecated_dir,
        gate_score=gate_score,
    )
    console.print(f"\n  [green]Graduated[/green] '{result.skill_name}' → {result.installed_path}")
    if result.branch:
        console.print(f"  Branch: {result.branch}")
    if result.archived_originals:
        console.print(f"  Archived merged originals: {', '.join(result.archived_originals)}")
    console.print()


@click.command("reject")
@click.argument("candidate_id")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Load a specific config TOML file.")
def reject_cmd(candidate_id, config_path):
    """Mark a candidate rejected (no change to the library)."""
    from src.cli.config import load_config
    from src.continuous import CandidateStore

    cfg = load_config(config_path=config_path)
    store = CandidateStore(cfg.continuous_candidates_dir)
    updated = store.set_status(candidate_id, "rejected")
    if updated is None:
        console.print(f"[red]Error:[/red] no candidate '{candidate_id}'.")
        raise SystemExit(1)
    console.print(f"  Rejected '{candidate_id}'.")
