from __future__ import annotations

import subprocess
from collections.abc import Iterable


class ExperimentRunner:
    """Runs explicit experiment commands while keeping CLI orchestration out of trainers."""

    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def run(self, command: Iterable[str]) -> None:
        resolved = list(command)
        print("$ " + " ".join(resolved), flush=True)
        if not self.dry_run:
            subprocess.run(resolved, check=True)
