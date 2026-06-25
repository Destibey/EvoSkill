"""Continuous evolution: learn skills from real agent usage traces.

Phase 0 surface — trace ingestion:

* `TaskEpisode` / `ActionStep` / `ToolCall` — normalized usage unit.
* `OutcomeSignal` / `Outcome` / `SignalKind` — how an attempt's success is judged.
* Readers: `HarborTrajectoryReader` (primary, ATIF), `GooseRawReader` (fallback),
  `JsonlReader` (generic).
* `TraceCollector` + `TraceCursor` — dedup + watermark across continuous ticks.

See `plan.md` for the full continuous-evolution design.
"""

from .episode import (
    ActionStep,
    Outcome,
    OutcomeSignal,
    SignalKind,
    TaskEpisode,
    ToolCall,
)
from .signals import clamp01, no_signal, signal_from_reward
from .collector import (
    GooseRawReader,
    HarborTrajectoryReader,
    JsonlReader,
    TraceCollector,
    TraceCursor,
    TraceReader,
    TraceReadError,
    parse_atif_trajectory,
    read_reward_file,
)
from .cluster import EpisodeCluster, cluster_episodes
from .candidates import Candidate, CandidateStore, make_candidate_id
from .harvest import (
    HarvestResult,
    build_distiller_query,
    build_readers,
    distill_clusters,
    harvest,
    slugify_skill_name,
)
from .library import Skill, SkillLibrary, parse_skill_file
from .similarity import (
    EmbeddingCache,
    EmbeddingSimilarity,
    LexicalSimilarity,
    SimilarityBackend,
    make_openai_embedder,
    make_similarity_backend,
)
from .skill_stats import (
    SkillStats,
    SkillStatsStore,
    assign_credit,
    assign_credit_batch,
)
from .lifecycle import (
    DeprecationReport,
    DuplicateGroup,
    SkillMatch,
    archive_skill,
    build_merge_query,
    evaluate_deprecation,
    find_duplicates,
    propose_merge,
    restore_skill,
    select_skills,
)
from .gate import (
    EvalOutcome,
    GateEvaluator,
    GateTask,
    GateVerdict,
    SurrogateEvaluator,
    build_replay_buffer,
    build_surrogate_query,
    run_gate,
)
from .graduation import GraduationResult, GraduationSafetyError, graduate, install_skill
from .loop import (
    CostMeter,
    MeteredAgent,
    TickConfig,
    TickReport,
    run_tick,
    run_watch_loop,
)

__all__ = [
    # episode model
    "ActionStep",
    "Outcome",
    "OutcomeSignal",
    "SignalKind",
    "TaskEpisode",
    "ToolCall",
    # signals
    "clamp01",
    "no_signal",
    "signal_from_reward",
    # collector
    "GooseRawReader",
    "HarborTrajectoryReader",
    "JsonlReader",
    "TraceCollector",
    "TraceCursor",
    "TraceReader",
    "TraceReadError",
    "parse_atif_trajectory",
    "read_reward_file",
    # clustering
    "EpisodeCluster",
    "cluster_episodes",
    # candidates
    "Candidate",
    "CandidateStore",
    "make_candidate_id",
    # harvest
    "HarvestResult",
    "build_distiller_query",
    "build_readers",
    "distill_clusters",
    "harvest",
    "slugify_skill_name",
    # library
    "Skill",
    "SkillLibrary",
    "parse_skill_file",
    # similarity
    "SimilarityBackend",
    "LexicalSimilarity",
    "EmbeddingSimilarity",
    "EmbeddingCache",
    "make_openai_embedder",
    "make_similarity_backend",
    # skill stats / credit
    "SkillStats",
    "SkillStatsStore",
    "assign_credit",
    "assign_credit_batch",
    # lifecycle
    "DuplicateGroup",
    "SkillMatch",
    "DeprecationReport",
    "find_duplicates",
    "select_skills",
    "evaluate_deprecation",
    "archive_skill",
    "restore_skill",
    "build_merge_query",
    "propose_merge",
    # gate
    "GateTask",
    "GateVerdict",
    "EvalOutcome",
    "GateEvaluator",
    "SurrogateEvaluator",
    "build_replay_buffer",
    "build_surrogate_query",
    "run_gate",
    # graduation
    "GraduationResult",
    "GraduationSafetyError",
    "graduate",
    "install_skill",
    # loop / watch tick
    "CostMeter",
    "MeteredAgent",
    "TickConfig",
    "TickReport",
    "run_tick",
    "run_watch_loop",
]
