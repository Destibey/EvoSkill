"""Load and validate .evoskill/config.toml + task.md."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.harness.model_aliases import (
    HarnessName,
    default_model_for_harness,
    normalize_harness_model,
)

def _docker_path_overrides() -> dict[str, str]:
    """Read path overrides injected by the Docker launcher."""
    raw = os.environ.get("EVOSKILL_PATH_OVERRIDES", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}

EVOSKILL_DIR = '.evoskill'


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


@dataclass
class HarnessConfig:
    name: HarnessName = 'claude'
    model: str | None = field(default_factory=lambda: default_model_for_harness("claude"))
    data_dirs: list[str] = field(default_factory=list)
    timeout_seconds: int = 1200
    max_retries: int = 3


@dataclass
class EvolutionConfig:
    mode: Literal['skill_only', 'prompt_only'] = 'skill_only'
    iterations: int = 7
    frontier_size: int = 3
    concurrency: int = 4
    no_improvement_limit: int = 5
    failure_samples: int = 3


@dataclass
class DatasetConfig:
    source: Literal['csv', 'harbor'] = 'csv'
    path: str = '/absolute/path/to/questions.csv'
    question_column: str = 'question'
    ground_truth_column: str = 'ground_truth'
    category_column: str | None = None
    train_ratio: float = 0.18
    val_ratio: float = 0.12
    # Harbor-specific (only used when source == "harbor")
    harbor_tasks_root: str = ''       # path to a downloaded harbor dataset dir
    harbor_limit: int | None = None   # max tasks to include
    harbor_include: list[str] = field(default_factory=list)  # glob patterns on task id
    harbor_exclude: list[str] = field(default_factory=list)
    harbor_difficulty: list[str] = field(default_factory=list)  # e.g. ["easy"] — empty = no filter


@dataclass
class HarborConfig:
    """Run the base agent inside Harbor sandboxes instead of via the LLM SDK.

    When enabled, the base agent's run() invokes `harbor run -p <task>` with
    EvoSkill's evolved skills mounted, and reads the reward from reward.txt.
    Proposer/generator agents continue to use harness.name's SDK normally.
    """
    enabled: bool = False
    inner_agent: str = 'claude-code'           # one of harbor's registered agents
    inner_model: str = 'anthropic/claude-sonnet-4-5'
    env: Literal['docker', 'daytona', 'modal', 'e2b', 'runloop'] = 'docker'
    n_concurrent: int = 1                      # parallel trials per harbor invocation
    timeout_multiplier: float = 1.0
    jobs_dir: str = ''                         # where to drop harbor jobs/<id>/ output (default: <project>/.evoskill/harbor_jobs)
    container_skills_path: str = '/skills'     # path inside container we mount evolved skills to
    extra_args: list[str] = field(default_factory=list)  # passthrough to harbor run


@dataclass
class LifecycleConfig:
    """Skill-library lifecycle management ([continuous.lifecycle])."""
    similarity_backend: Literal['embedding', 'lexical'] = 'embedding'
    embedding_provider: str = 'openai'          # OpenAI-compatible; else falls back to lexical
    embedding_model: str = 'text-embedding-3-small'
    dedupe_similarity: float = 0.88             # cosine threshold for "redundant"
    deprecation_baseline: float = 0.0           # contribution floor
    deprecation_strikes: int = 3                # windows below floor before retiring
    retrieval_top_k: int = 6                    # skills to surface per task


@dataclass
class ContinuousConfig:
    """Continuous evolution: learn skills from real usage traces.

    Phase 1 (harvest) uses the harvest-relevant fields below; Phase 2 adds the
    nested `lifecycle` section. Later phases add graduation/signal settings here.
    """
    enabled: bool = False
    trace_sources: list[str] = field(default_factory=lambda: ['harbor'])  # harbor | goose | jsonl
    traces_root: str = ''                 # override; default <project>/.evoskill/harbor_jobs
    jsonl_path: str = ''                  # path for the 'jsonl' source
    harvest_window: int = 200             # max episodes per harvest
    min_cluster_size: int = 3             # episodes needed to justify a candidate
    similarity_threshold: float = 0.3     # cosine threshold for clustering
    max_candidates: int | None = None     # cap clusters distilled per harvest
    focus: Literal['failure', 'success', 'both'] = 'failure'
    distiller_model: str | None = None    # defaults to harness.model
    concurrency: int = 4                  # parallel distiller calls
    success_threshold: float = 1.0        # verifier reward >= this counts as success
    # Graduation / gate (Phase 3)
    graduation_mode: Literal['review', 'auto'] = 'review'  # 'review' = human applies; 'auto' = gate applies
    graduation_threshold: float = 0.6     # min surrogate gate score to pass
    shadow_eval_size: int = 10            # held-out tasks shown to the verifier
    max_graduations_per_window: int = 2   # rate limit (used by the watch daemon)
    surrogate_model: str | None = None    # defaults to harness.model
    # Watch daemon (Phase 4)
    poll_interval_sec: int = 600          # seconds between watch ticks
    cost_ceiling_usd_per_tick: float = 0.0  # 0 = unlimited; else stop a tick past this spend
    auto_deprecate: bool = False          # in auto mode, archive skills past the strike limit
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)


@dataclass
class ScorerConfig:
    type: Literal['exact', 'multi_tolerance', 'llm', 'script', 'harbor'] = 'multi_tolerance'
    rubric: str | None = None
    model: str | None = None
    provider: str | None = None
    command: str | None = None


@dataclass
class DaytonaConfig:
    api_key: str | None = None
    image: str = ''
    cpu: int = 4
    memory: int = 8        # GB
    disk: int = 10         # GB
    timeout: int = 0       # 0 = no auto-stop


@dataclass
class DownloadConfig:
    all_branches: bool = False
    cache: bool = False
    reports: bool = True
    feedback_history: bool = False


_VALID_REMOTE_TARGETS = ('daytona',)


@dataclass
class RemoteConfig:
    target: str = 'daytona'
    daytona: DaytonaConfig | None = None
    download: DownloadConfig = field(default_factory=DownloadConfig)


@dataclass
class ProjectConfig:
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    remote: RemoteConfig | None = None
    harbor: HarborConfig = field(default_factory=HarborConfig)
    continuous: ContinuousConfig = field(default_factory=ContinuousConfig)
    execution: str = 'local'  # 'local', 'docker', or 'daytona'
    project_root: Path = field(default_factory=Path.cwd)
    task_description: str = ''
    task_constraints: str = ''

    @property
    def evoskill_dir(self) -> Path:
        return self.project_root / EVOSKILL_DIR

    @property
    def dataset_path(self) -> Path:
        """Return the dataset CSV path, with container override and relative path support."""
        override = _docker_path_overrides().get("dataset_path")
        if override:
            return Path(override)
        path = Path(self.dataset.path)
        return path if path.is_absolute() else self.project_root / path

    @property
    def harbor_tasks_root_path(self) -> Path:
        """Return the harbor tasks root, with container override and relative path support."""
        override = _docker_path_overrides().get("harbor_tasks_root")
        if override:
            return Path(override)
        path = Path(self.dataset.harbor_tasks_root)
        return path if path.is_absolute() else self.project_root / path

    @property
    def harbor_jobs_dir(self) -> Path:
        """Where Harbor trial output (incl. trajectory.json) is written."""
        if self.harbor.jobs_dir:
            path = Path(self.harbor.jobs_dir)
            return path if path.is_absolute() else self.project_root / path
        return self.evoskill_dir / 'harbor_jobs'

    @property
    def continuous_traces_root(self) -> Path:
        """Default trace root for continuous harvest (Harbor job output)."""
        if self.continuous.traces_root:
            return _resolve_project_path(self.project_root, self.continuous.traces_root)
        return self.harbor_jobs_dir

    @property
    def continuous_jsonl_path(self) -> Path | None:
        """Configured JSONL trace path, resolved relative to the project root."""
        if not self.continuous.jsonl_path:
            return None
        return _resolve_project_path(self.project_root, self.continuous.jsonl_path)

    @property
    def continuous_complaints_path(self) -> Path:
        """Default local trace file for operator complaint feedback."""
        return self.evoskill_dir / 'continuous' / 'complaints.jsonl'

    @property
    def continuous_candidates_dir(self) -> Path:
        """Where harvested candidate skills are buffered for review."""
        return self.evoskill_dir / 'continuous' / 'candidates'

    @property
    def skills_dir(self) -> Path:
        """The live skill library."""
        return self.project_root / '.claude' / 'skills'

    @property
    def continuous_skill_stats_path(self) -> Path:
        return self.evoskill_dir / 'continuous' / 'skill_stats.json'

    @property
    def continuous_embeddings_cache_path(self) -> Path:
        return self.evoskill_dir / 'continuous' / 'embeddings_cache.json'

    @property
    def continuous_deprecated_dir(self) -> Path:
        return self.evoskill_dir / 'continuous' / 'deprecated'


def _find_project_root(start: Path | None = None) -> Path | None:
    """Walk up from start looking for a .evoskill/ directory."""
    current = Path.cwd() if start is None else start
    for parent in [current, *current.parents]:
        if (parent / EVOSKILL_DIR).exists():
            return parent
    return None


def _parse_task_md(text: str) -> tuple[str, str]:
    """Split task.md into (description, constraints) at the --- separator."""
    parts = text.split('\n---\n', maxsplit=1)
    description = parts[0].strip()
    constraints = parts[1].strip() if len(parts) > 1 else ''
    return description, constraints


def _resolve_config_override(config_path: Path) -> tuple[Path, Path]:
    """Resolve an explicit config path and the project root it belongs to."""
    path = config_path.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()

    root = _find_project_root(path.parent)
    if root is None:
        import sys
        print(
            f"Error: no .evoskill/ directory found above config file {path}."
        )
        sys.exit(1)

    return root, path


def load_config(
    start: Path | None = None,
    config_path: Path | None = None,
) -> ProjectConfig:
    """Find and load the project config. Exits with a message if not found."""
    if config_path is not None:
        root, config_path = _resolve_config_override(config_path)
    else:
        root = _find_project_root(start)
        if root is None:
            import sys
            print("Error: no .evoskill/ directory found. Run 'evoskill init' first.")
            sys.exit(1)
        config_path = root / EVOSKILL_DIR / 'config.toml'

    if not config_path.exists():
        import sys
        print(f"Error: config file not found at {config_path}.")
        sys.exit(1)

    with open(config_path, 'rb') as f:
        raw = tomllib.load(f)

    harness_raw = dict(raw.get('harness', {}))
    harness_name = harness_raw.get('name', 'claude')
    harness_raw['model'] = normalize_harness_model(harness_name, harness_raw.get('model'))

    # Docker path overrides for data_dirs
    overrides = _docker_path_overrides()
    if "data_dirs" in overrides:
        harness_raw['data_dirs'] = [d.strip() for d in overrides["data_dirs"].split(",") if d.strip()]

    harness = HarnessConfig(**harness_raw)
    evolution = EvolutionConfig(**raw.get('evolution', {}))
    dataset = DatasetConfig(**raw.get('dataset', {}))
    scorer = ScorerConfig(**raw.get('scorer', {}))

    # Parse remote config
    remote: RemoteConfig | None = None
    remote_raw = raw.get('remote')
    if remote_raw is not None:
        target = remote_raw.get('target', 'daytona')
        if target not in _VALID_REMOTE_TARGETS:
            raise ValueError(
                f"Unsupported remote target '{target}'. "
                f"Valid targets: {', '.join(_VALID_REMOTE_TARGETS)}"
            )

        daytona_cfg: DaytonaConfig | None = None
        if 'daytona' in remote_raw:
            daytona_cfg = DaytonaConfig(**remote_raw['daytona'])

        # Daytona API key: toml takes precedence, fall back to env var
        if daytona_cfg is not None and not daytona_cfg.api_key:
            env_key = os.environ.get('DAYTONA_API_KEY')
            if env_key:
                daytona_cfg.api_key = env_key
        elif target == 'daytona' and daytona_cfg is None:
            # target is daytona but no [remote.daytona] section — create with defaults
            env_key = os.environ.get('DAYTONA_API_KEY')
            daytona_cfg = DaytonaConfig(api_key=env_key)

        download_cfg = DownloadConfig(**remote_raw.get('download', {}))

        remote = RemoteConfig(
            target=target,
            daytona=daytona_cfg,
            download=download_cfg,
        )

    task_path = root / EVOSKILL_DIR / 'task.md'
    description, constraints = _parse_task_md(task_path.read_text()) if task_path.exists() else ('', '')

    execution = raw.get('execution', 'local')

    # Parse [harbor] section (optional)
    harbor_raw = dict(raw.get('harbor', {}))
    harbor = HarborConfig(**harbor_raw) if harbor_raw else HarborConfig()

    # Parse [continuous] section (optional), including the nested
    # [continuous.lifecycle] table.
    continuous_raw = dict(raw.get('continuous', {}))
    lifecycle_raw = continuous_raw.pop('lifecycle', None)
    continuous = ContinuousConfig(**continuous_raw) if continuous_raw else ContinuousConfig()
    if lifecycle_raw:
        continuous.lifecycle = LifecycleConfig(**lifecycle_raw)

    return ProjectConfig(
        harness=harness,
        evolution=evolution,
        dataset=dataset,
        scorer=scorer,
        remote=remote,
        harbor=harbor,
        continuous=continuous,
        execution=execution,
        project_root=root,
        task_description=description,
        task_constraints=constraints,
    )
