"""evoskill skill-eval - export an agent-skills-eval packet for a candidate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command("skill-eval")
@click.argument("candidate_id")
@click.option("--output", "output_dir", type=click.Path(path_type=Path), default=None,
              help="Directory to write the eval packet. Default: .evoskill/continuous/external-evals/<id>.")
@click.option("--traces", "traces_root", type=click.Path(path_type=Path), default=None,
              help="Trace root to replay (default: configured harbor jobs dir).")
@click.option("--jsonl", "jsonl_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="JSONL trace file to replay. Relative paths are project-relative.")
@click.option("--source", "sources", multiple=True,
              type=click.Choice(["harbor", "goose", "jsonl"]),
              help="Trace source(s) to read. Repeatable. Default: config.")
@click.option("--window", type=int, default=None, help="Max episodes to collect.")
@click.option("--max-evals", type=int, default=None, help="Max held-out replay tasks to export.")
@click.option("--target", "target_model", default=None, help="agent-skills-eval target model.")
@click.option("--judge", "judge_model", default=None, help="agent-skills-eval judge model.")
@click.option("--base-url", default="https://api.openai.com/v1",
              help="OpenAI-compatible base URL for agent-skills-eval.")
@click.option("--api-key-env", default="OPENAI_API_KEY",
              help="Environment variable name for the external evaluator API key.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Load a specific config TOML file.")
def skill_eval_cmd(
    candidate_id,
    output_dir,
    traces_root,
    jsonl_path,
    sources,
    window,
    max_evals,
    target_model,
    judge_model,
    base_url,
    api_key_env,
    config_path,
):
    """Export a no-write agent-skills-eval packet for independent review."""
    from src.cli.config import load_config
    from src.continuous import CandidateStore, TraceCollector, build_readers, build_replay_buffer

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
    max_evals = max_evals if max_evals is not None else cont.shadow_eval_size
    target_model = target_model or cont.surrogate_model or cfg.harness.model or "gpt-4o-mini"
    judge_model = judge_model or target_model

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
    replay = build_replay_buffer(episodes, candidate, size=max_evals)
    if not replay:
        console.print(
            "[red]Error:[/red] no held-out replay tasks for external eval export.\n"
            "  Add traces outside the candidate's source episodes before exporting."
        )
        raise SystemExit(1)

    out = _resolve_output_dir(cfg, output_dir, candidate.candidate_id)
    skill_dir = out / "skills" / candidate.skill_name
    evals_dir = skill_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)

    (skill_dir / "SKILL.md").write_text(candidate.skill_markdown)
    (evals_dir / "evals.json").write_text(
        json.dumps(_evals_payload(candidate, replay), indent=2) + "\n"
    )
    (out / "agent-skills-eval.yaml").write_text(
        _agent_skills_eval_yaml(candidate, target_model, judge_model, base_url, api_key_env)
    )
    (out / "evoskill-provenance.json").write_text(
        json.dumps(_provenance(candidate, replay, src_list, root, jsonl), indent=2) + "\n"
    )
    (out / "README.md").write_text(_readme(candidate, api_key_env))

    console.print(f"\n  Exported agent-skills-eval packet: [bold]{out}[/bold]")
    console.print(f"  Skill: {candidate.skill_name}")
    console.print(f"  Evals: {len(replay)} held-out replay task(s)")
    console.print(f"  Run: {api_key_env}=... npx agent-skills-eval --config agent-skills-eval.yaml\n")


def _evals_payload(candidate, replay) -> dict:
    evals = []
    for i, episode in enumerate(replay, start=1):
        eval_id = _safe_eval_id(episode.task_id or episode.episode_id or f"eval-{i}")
        expected = _expected_output(candidate)
        evals.append(
            {
                "id": eval_id,
                "name": episode.task_id or episode.episode_id or f"held-out replay {i}",
                "prompt": episode.task_text,
                "expected_output": expected,
                "assertions": [
                    "The output directly satisfies the user's task.",
                    expected,
                ],
            }
        )
    return {"skill_name": candidate.skill_name, "evals": evals}


def _expected_output(candidate) -> str:
    pattern = (candidate.target_pattern or "").strip()
    if pattern:
        return f"The output should address the recurring failure pattern: {pattern}"
    return "The output should avoid the failure pattern that produced this candidate skill."


def _agent_skills_eval_yaml(candidate, target_model, judge_model, base_url, api_key_env) -> str:
    return "\n".join(
        [
            "root: ./skills",
            "workspace: ./agent-skills-workspace",
            "baseline: true",
            f"target: {_yaml_scalar(target_model)}",
            f"judge: {_yaml_scalar(judge_model)}",
            f"baseUrl: {_yaml_scalar(base_url)}",
            f"apiKeyEnv: {_yaml_scalar(api_key_env)}",
            "include:",
            '  - "skills/**"',
            "concurrency: 2",
            "layout: iteration",
            "strict: true",
            "report:",
            "  enabled: true",
            f"  title: {_yaml_scalar(f'EvoSkill Candidate Evaluation - {candidate.skill_name}')}",
            "logging:",
            "  format: pretty",
            "  verbose: false",
            "  color: auto",
            "targetParams:",
            "  temperature: 0",
            "judgeParams:",
            "  temperature: 0",
            "",
        ]
    )


def _provenance(candidate, replay, sources, traces_root, jsonl_path) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "evoskill skill-eval",
        "external_tool": "agent-skills-eval",
        "candidate_id": candidate.candidate_id,
        "skill_name": candidate.skill_name,
        "candidate_source": candidate.source,
        "candidate_episode_ids": candidate.episode_ids,
        "replay_episode_ids": [e.episode_id for e in replay],
        "trace_sources": sources,
        "traces_root": traces_root,
        "jsonl_path": str(jsonl_path) if jsonl_path else None,
        "non_claims": [
            "This packet does not run agent-skills-eval.",
            "This packet does not install or graduate the candidate skill.",
            "The generated assertions are bootstrap checks and should be edited for high-stakes acceptance.",
        ],
    }


def _readme(candidate, api_key_env: str) -> str:
    return f"""# EvoSkill external skill eval packet

Candidate: `{candidate.candidate_id}` / `{candidate.skill_name}`

This directory is a no-write bridge to `agent-skills-eval`.

Run from this directory:

```bash
{api_key_env}=... npx agent-skills-eval --config agent-skills-eval.yaml
```

Expected artifacts are written under `agent-skills-workspace/`, including the
HTML report and JSON/JSONL evidence generated by `agent-skills-eval`.

This packet is review-only. It does not graduate the candidate and does not
write to the live skill library.
"""


def _resolve_output_dir(cfg, output_dir: Path | None, candidate_id: str) -> Path:
    if output_dir is None:
        return cfg.project_root / ".evoskill" / "continuous" / "external-evals" / candidate_id
    path = output_dir.expanduser()
    return path if path.is_absolute() else cfg.project_root / path


def _resolve_jsonl_path(cfg, jsonl_path: Path | None) -> Path | None:
    if jsonl_path is None:
        return cfg.continuous_jsonl_path
    path = jsonl_path.expanduser()
    return path if path.is_absolute() else cfg.project_root / path


def _safe_eval_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value.strip().lower())
    return safe.strip("-") or "eval"


def _yaml_scalar(value: str) -> str:
    return json.dumps(str(value))
