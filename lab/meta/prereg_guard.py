"""Code-enforced separation of development and evaluation.

`assert_preregistered(spec_file)` refuses to continue unless the pre-registration file is tracked by git, has no
uncommitted changes, and is already part of a commit. Benchmarks call it before scoring held-out tasks, so the
criteria cannot be edited after seeing a held-out score without leaving a commit that shows it.
It does not prove the criteria were not changed in an earlier commit; the commit history is the evidence.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class NotPreregistered(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def assert_preregistered(spec_file: Path, repo: Path | None = None) -> str:
    """Return the hash of the last commit that touched `spec_file`, or raise NotPreregistered."""
    spec_file = Path(spec_file).resolve()
    repo = Path(repo).resolve() if repo else spec_file.parent
    top = _git(repo, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        raise NotPreregistered("not inside a git repository: pre-registration cannot be verified")
    repo = Path(top.stdout.strip()).resolve()
    rel = str(spec_file.relative_to(repo))
    if _git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
        raise NotPreregistered(f"{rel} is not tracked by git")
    if _git(repo, "status", "--porcelain", "--", rel).stdout.strip():
        raise NotPreregistered(f"{rel} has uncommitted changes")
    h = _git(repo, "log", "-1", "--format=%H", "--", rel).stdout.strip()
    if not h:
        raise NotPreregistered(f"{rel} is in no commit")
    return h
