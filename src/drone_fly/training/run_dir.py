"""Run folders: ``artifacts/training/<drone-id>/<run-id>/`` (``artifacts/**`` is gitignored).

::

    <run-id>/                     run-id = <UTC stamp>-<config name>
      config.yaml                 the resolved config (incl. contract_version)
      drone.yaml                  byte copy of the profile; drone.sha256 its hash
      checkpoints/
        model_readout.pt          stripped readout (loads via inference.load_policy)
        model_readout.contract.json
        model_readout_it<N>.pt    periodic checkpoints (+ their sidecars)
      logs/
        train.log  metrics.jsonl  episodes.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from drone_fly.sim.action import CONTRACT_VERSION, ActionSpec, write_sidecar

CHECKPOINT_NAME = "model_readout.pt"


def default_root() -> Path:
    from drone_fly.harness.config import find_repo_root

    root = find_repo_root() or Path.cwd()
    return root / "artifacts" / "training"


class RunDir:
    def __init__(
        self,
        config: Any,
        profile: Any,
        *,
        root: str | Path | None = None,
        run_id: str | None = None,
    ) -> None:
        self.run_id = run_id or (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{config.name}")
        self.path = Path(root or default_root()) / profile.id / self.run_id
        self.checkpoints = self.path / "checkpoints"
        self.logs = self.path / "logs"
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.profile = profile
        snap = config.snapshot()
        snap["run_id"] = self.run_id
        (self.path / "config.yaml").write_text(
            yaml.safe_dump(snap, sort_keys=False), encoding="utf-8"
        )
        if profile.path:
            src = Path(profile.path)
            shutil.copyfile(src, self.path / "drone.yaml")
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            (self.path / "drone.sha256").write_text(f"{digest}  drone.yaml\n", encoding="utf-8")
        self.log_handler = logging.FileHandler(self.logs / "train.log", encoding="utf-8")
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    def save_checkpoint(
        self, state: dict[str, Any], spec: ActionSpec, *, extra_dim: int, it: int | None = None
    ) -> Path:
        import torch

        if spec.version == CONTRACT_VERSION and spec.max_g is None:
            spec = replace(spec, max_g=float(self.profile.max_g))  # the training envelope
        name = CHECKPOINT_NAME if it is None else f"model_readout_it{it}.pt"
        path = self.checkpoints / name
        torch.save(state, path)
        write_sidecar(
            path,
            spec,
            extra_dim=extra_dim,
            actor=self.config.actor.kind,
            drone=self.profile.id,
            run_id=self.run_id,
        )
        return path

    def metrics(self, row: dict[str, Any]) -> None:
        with open(self.logs / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def episodes(self, columns: tuple[str, ...], rows: np.ndarray) -> None:
        path = self.logs / "episodes.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(columns)
            for r in rows:
                w.writerow([f"{x:.6g}" for x in r])
