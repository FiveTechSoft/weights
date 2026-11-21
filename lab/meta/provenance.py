"""Link an evaluation to the pre-registered protocol and to the data and code it ran on (hashes only).

This records WHICH protocol text, data and code a result belongs to, so a later change is visible. It does not
seal the held-out set and does not stop anyone from looking at it: whoever has the scores file can read it.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_at_commit(repo: Path, commit: str, rel: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{rel}"], capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"{rel} does not exist at {commit}")
    return hashlib.sha256(out.stdout).hexdigest()


def provenance(repo: Path, protocol_rel: str, protocol_commit: str, data_rels, code_rels, results_rels) -> dict:
    repo = Path(repo)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "protocol_file": protocol_rel,
        "protocol_commit": protocol_commit,
        "protocol_sha256_at_commit": sha256_at_commit(repo, protocol_commit, protocol_rel),
        "protocol_sha256_now": sha256_file(repo / protocol_rel),
        "data_sha256": {r: sha256_file(repo / r) for r in data_rels},
        "code_sha256": {r: sha256_file(repo / r) for r in code_rels},
        "results_sha256": {r: sha256_file(repo / r) for r in results_rels},
        "repo_head_when_recorded": head,
        "limits": "hashes show which text/data/code a result belongs to; they do not seal the held-out set or "
                  "prove it was not inspected before the protocol was committed",
    }
