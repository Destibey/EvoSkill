"""Graduate a passing candidate into the live skill library.

Graduation is the only step that changes the live library, so it is deliberate
and auditable:

1. Install the candidate's `SKILL.md` into `.claude/skills/<name>/`.
2. For a *merge* candidate, archive the originals it replaces (reversibly).
3. Require a `ProgramManager` by default and snapshot the change as a
   `program/*` git branch — so every graduation is versioned and revertible,
   exactly like the batch loop's programs. (Branched from the current HEAD; it
   does NOT join the accuracy-based frontier, so continuous graduations never
   pollute the main loop's score ranking.)
4. Mark the candidate `graduated` in the buffer.

The `ProgramManager` is injected so the filesystem behaviour is unit-testable
without git, while a real-repo integration test exercises the branch path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .candidates import Candidate, CandidateStore
from .lifecycle import archive_skill


class GraduationSafetyError(RuntimeError):
    """Raised when graduation would write a live skill without rollback evidence."""


def install_skill(skills_dir: str | Path, skill_name: str, skill_markdown: str) -> Path:
    """Write a skill into the live library at `<skills_dir>/<name>/SKILL.md`."""
    target = Path(skills_dir) / skill_name
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(skill_markdown)
    return target


@dataclass
class GraduationResult:
    candidate_id: str
    skill_name: str
    installed_path: str
    branch: str | None = None
    archived_originals: list[str] = field(default_factory=list)


def _archive_merge_originals(
    candidate: Candidate, skills_dir: Path, archive_dir: str | Path | None
) -> list[str]:
    """For a merge candidate, archive the skills it supersedes. Best-effort."""
    if candidate.source != "merge" or archive_dir is None:
        return []
    archived: list[str] = []
    for original in candidate.extra.get("merged_from", []) or []:
        if original == candidate.skill_name:
            continue  # never archive the freshly-installed merged skill
        try:
            archive_skill(skills_dir, original, archive_dir)
            archived.append(original)
        except FileNotFoundError:
            continue
    return archived


def graduate(
    candidate: Candidate,
    *,
    skills_dir: str | Path,
    store: CandidateStore,
    manager: Any | None = None,
    archive_dir: str | Path | None = None,
    branch_prefix: str = "iter-skill",
    gate_score: float | None = None,
    require_snapshot: bool = True,
) -> GraduationResult:
    """Install a candidate skill and (optionally) version it as a `program/*` branch.

    Args:
        candidate: the candidate to graduate.
        skills_dir: the live `.claude/skills/` directory.
        store: candidate buffer (its status is set to "graduated").
        manager: a ProgramManager-like object; if given, the change is committed
            to a new `program/*` branch (branched from current HEAD).
        archive_dir: where merge-superseded skills are archived (reversible).
        gate_score: the gate score, recorded in branch metadata for audit.
        require_snapshot: when True, require a ProgramManager-backed program
            branch before touching the live skill library.

    Returns:
        GraduationResult describing what was installed/archived/branched.
    """
    skills_dir = Path(skills_dir)
    if require_snapshot and manager is None:
        raise GraduationSafetyError(
            "graduation requires a program/* snapshot before live skill writeback"
        )

    installed = install_skill(skills_dir, candidate.skill_name, candidate.skill_markdown)
    archived = _archive_merge_originals(candidate, skills_dir, archive_dir)

    branch: str | None = None
    if manager is not None:
        branch_name = f"{branch_prefix}-{candidate.candidate_id}"
        config = manager.get_current().mutate(
            name=branch_name,
            metadata={
                "graduated_from": candidate.candidate_id,
                "source": candidate.source,
                "gate_score": gate_score,
                "continuous": True,
            },
        )
        # parent=None → branch from current HEAD; the installed (untracked) skill
        # is carried onto the new branch and committed by create_program.
        branch = manager.create_program(branch_name, config, parent=None)

    store.set_status(candidate.candidate_id, "graduated")
    return GraduationResult(
        candidate_id=candidate.candidate_id,
        skill_name=candidate.skill_name,
        installed_path=str(installed),
        branch=branch,
        archived_originals=archived,
    )
