"""Tests for graduation: skill install, branch versioning, merge archival."""

from __future__ import annotations

import subprocess

import pytest

from src.continuous.candidates import Candidate, CandidateStore
from src.continuous.graduation import GraduationSafetyError, graduate, install_skill


def _candidate(**kw) -> Candidate:
    base = dict(
        candidate_id="units-abc",
        skill_name="preserve-units",
        skill_markdown="---\nname: preserve-units\ndescription: keep units\n---\nAlways include units.",
        episode_ids=["e1", "e2"],
        cluster_size=2,
    )
    base.update(kw)
    return Candidate(**base)


class _FakeManager:
    """Records create_program calls; returns a branch name."""

    def __init__(self):
        self.created = []

    def get_current(self):
        from src.registry.models import ProgramConfig
        return ProgramConfig(name="base", system_prompt={"type": "preset", "preset": "claude_code"})

    def create_program(self, name, config, parent=None):
        self.created.append((name, parent, config.metadata))
        return f"program/{name}"


class TestInstallSkill:
    def test_writes_skill_md(self, tmp_path):
        path = install_skill(tmp_path / "skills", "my-skill", "---\nname: my-skill\n---\nbody")
        assert (path / "SKILL.md").read_text().startswith("---")


class TestGraduateNoManager:
    def test_default_requires_snapshot_before_writing(self, tmp_path):
        skills = tmp_path / ".claude" / "skills"
        store = CandidateStore(tmp_path / "cands")
        store.save(_candidate())
        with pytest.raises(GraduationSafetyError):
            graduate(_candidate(), skills_dir=skills, store=store, manager=None, gate_score=0.9)

        assert not (skills / "preserve-units" / "SKILL.md").exists()
        assert store.get("units-abc").status == "pending"

    def test_installs_and_marks(self, tmp_path):
        skills = tmp_path / ".claude" / "skills"
        store = CandidateStore(tmp_path / "cands")
        store.save(_candidate())
        result = graduate(
            _candidate(),
            skills_dir=skills,
            store=store,
            manager=None,
            gate_score=0.9,
            require_snapshot=False,
        )
        assert (skills / "preserve-units" / "SKILL.md").is_file()
        assert result.branch is None
        assert store.get("units-abc").status == "graduated"


class TestGraduateWithManager:
    def test_creates_branch_with_metadata(self, tmp_path):
        skills = tmp_path / ".claude" / "skills"
        store = CandidateStore(tmp_path / "cands")
        store.save(_candidate())
        mgr = _FakeManager()
        result = graduate(_candidate(), skills_dir=skills, store=store, manager=mgr, gate_score=0.85)
        assert result.branch == "program/iter-skill-units-abc"
        assert len(mgr.created) == 1
        name, parent, metadata = mgr.created[0]
        assert name == "iter-skill-units-abc"
        assert parent is None  # branches from current HEAD
        assert metadata["gate_score"] == 0.85
        assert metadata["continuous"] is True


class TestMergeArchival:
    def test_merge_archives_originals(self, tmp_path):
        skills = tmp_path / ".claude" / "skills"
        # two originals exist in the live library
        for name in ("preserve-units", "keep-units"):
            (skills / name).mkdir(parents=True)
            (skills / name / "SKILL.md").write_text(f"---\nname: {name}\n---\nx")
        archive = tmp_path / "deprecated"
        store = CandidateStore(tmp_path / "cands")
        merged = _candidate(
            candidate_id="merged-x", skill_name="units-merged",
            source="merge", extra={"merged_from": ["preserve-units", "keep-units"]},
        )
        store.save(merged)
        result = graduate(
            merged,
            skills_dir=skills,
            store=store,
            manager=None,
            archive_dir=archive,
            require_snapshot=False,
        )
        assert (skills / "units-merged" / "SKILL.md").is_file()       # merged installed
        assert set(result.archived_originals) == {"preserve-units", "keep-units"}
        assert (archive / "preserve-units").is_dir()                  # originals archived
        assert not (skills / "preserve-units").exists()


class TestGraduateRealGit:
    """Integration: graduation against a real git repo produces a program/* branch."""

    def _git(self, *args, cwd):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)

    def test_branch_created_and_skill_committed(self, tmp_path):
        self._git("init", "-q", cwd=tmp_path)
        self._git("config", "user.email", "t@t.co", cwd=tmp_path)
        self._git("config", "user.name", "t", cwd=tmp_path)
        (tmp_path / "README.md").write_text("x")
        self._git("add", "-A", cwd=tmp_path)
        self._git("commit", "-q", "-m", "init", cwd=tmp_path)

        from src.registry import ProgramManager
        from src.registry.models import ProgramConfig

        mgr = ProgramManager(cwd=tmp_path)
        mgr.create_program(
            "base",
            ProgramConfig(name="base", system_prompt={"type": "preset", "preset": "claude_code"},
                          allowed_tools=["Read"]),
        )
        store = CandidateStore(tmp_path / ".evoskill" / "continuous" / "candidates")
        store.save(_candidate())

        result = graduate(
            _candidate(), skills_dir=tmp_path / ".claude" / "skills",
            store=store, manager=mgr,
            archive_dir=tmp_path / ".evoskill" / "continuous" / "deprecated", gate_score=0.9,
        )
        assert result.branch == "program/iter-skill-units-abc"

        branches = subprocess.run(["git", "branch", "--list"], cwd=tmp_path,
                                   capture_output=True, text=True).stdout
        assert "program/iter-skill-units-abc" in branches

        tracked = subprocess.run(["git", "ls-files", ".claude/skills"], cwd=tmp_path,
                                 capture_output=True, text=True).stdout
        assert "preserve-units/SKILL.md" in tracked
