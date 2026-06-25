"""EvoSkill CLI entry point."""

from importlib import import_module

import click

_COMMAND_SPECS = {
    "init": ("src.cli.commands.init", "init_cmd", "Initialize a new EvoSkill project in the current directory."),
    "run": ("src.cli.commands.run", "run_cmd", "Run the self-improvement loop."),
    "eval": ("src.cli.commands.eval", "eval_cmd", "Evaluate the best skills on the validation set."),
    "complaint": ("src.cli.commands.complaint", "complaint_cmd", "Record operator dissatisfaction as a harvestable JSONL failure trace."),
    "harvest": ("src.cli.commands.harvest", "harvest_cmd", "Distill candidate skills from real usage traces."),
    "candidates": ("src.cli.commands.candidates", "candidates_cmd", "List/inspect harvested candidate skills."),
    "gate": ("src.cli.commands.gate", "gate_cmd", "Evaluate a candidate without live skill writeback."),
    "library": ("src.cli.commands.library", "library_cmd", "Inspect/curate the skill library (stats, duplicates, retrieval)."),
    "skill-eval": ("src.cli.commands.skill_eval", "skill_eval_cmd", "Export an agent-skills-eval packet for a candidate."),
    "graduate": ("src.cli.commands.graduate", "graduate_cmd", "Gate and graduate a candidate skill into the library."),
    "reject": ("src.cli.commands.graduate", "reject_cmd", "Mark a candidate skill rejected."),
    "watch": ("src.cli.commands.watch", "watch_cmd", "Run the continuous-evolution daemon (collect→distill→gate→graduate)."),
    "skills": ("src.cli.commands.skills", "skills_cmd", "List all skills learned so far."),
    "diff": ("src.cli.commands.diff", "diff_cmd", "Diff baseline vs best, or between two specific iterations."),
    "logs": ("src.cli.commands.logs", "logs_cmd", "Show recent run history."),
    "reset": ("src.cli.commands.reset", "reset_cmd", "Delete all program branches and frontier tags for a clean slate."),
    "remote": ("src.cli.commands.remote", "remote_group", "Manage remote EvoSkill runs."),
}


class LazyGroup(click.Group):
    def list_commands(self, ctx):
        return sorted(_COMMAND_SPECS)

    def get_command(self, ctx, cmd_name):
        spec = _COMMAND_SPECS.get(cmd_name)
        if spec is None:
            return None
        module_name, attr_name, _ = spec
        module = import_module(module_name)
        return getattr(module, attr_name)

    def format_commands(self, ctx, formatter):
        rows = [(name, spec[2]) for name, spec in sorted(_COMMAND_SPECS.items())]
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


@click.group(cls=LazyGroup)
def cli():
    """EvoSkill CLI."""
