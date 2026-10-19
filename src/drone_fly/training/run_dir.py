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

UC68 teacher runs (``drone-fly train teacher``) add::

      checkpoints/
        teacher.pt  teacher.contract.json         latest (every checkpoint_every_s + at the end)
        teacher_it<N>.pt (+ sidecar)              periodic (every checkpoint_every_iters)
        resume.pt                                 actor, critic, optimizer, counters, RNG states
      tracks/
        catalogue-<source>.yaml                   the catalogues the run selected from
        courses.json                              pool index -> id, source, split, map, scene, group
        files.sha256                              every TRAIN file loaded (XMLs, props, scenes with
                                                  their resolved source paths)
      logs/
        progress.json                             steps, fps, ETA (rewritten every log interval)
        tb/                                       TensorBoard, when importable
      eval/
        <split>.json  <split>.md                  ``drone-fly train eval`` reports

Every checkpoint / resume / progress file is written atomically (temp file + rename). ``--resume``
reopens a run without rewriting ``config.yaml``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from drone_fly.sim.action import (
    CONTRACT_VERSION,
    ActionSpec,
    SlewSpec,
    obs_contract_version,
    write_sidecar,
)

CHECKPOINT_NAME = "model_readout.pt"
TEACHER_NAME = "teacher.pt"
RESUME_NAME = "resume.pt"


def _slew_sidecar_extra(config: Any) -> dict[str, Any] | None:
    """UC-88: the ``action_filter`` sidecar key when the training course enabled the slew limiter,
    else ``None`` so the sidecar stays byte-identical for runs trained without it. Writing it into
    the checkpoint makes deploy == training by construction (the harness reads it back)."""
    course = getattr(config, "course", None)
    slew_time_s = getattr(course, "slew_time_s", None) if course is not None else None
    if slew_time_s is None:
        return None
    return {"action_filter": SlewSpec(float(slew_time_s)).as_dict()}


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
        reopen: Path | None = None,
    ) -> None:
        if reopen is not None:
            self.path = Path(reopen)
            self.run_id = self.path.name
        else:
            self.run_id = run_id or (
                datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{config.name}"
            )
            self.path = Path(root or default_root()) / profile.id / self.run_id
        self.checkpoints = self.path / "checkpoints"
        self.logs = self.path / "logs"
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.profile = profile
        self.log_handler = logging.FileHandler(self.logs / "train.log", encoding="utf-8")
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        if reopen is not None:
            return
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

    @classmethod
    def reopen(cls, path: str | Path, config: Any, profile: Any) -> RunDir:
        """An existing run folder (``--resume``): config.yaml and drone.yaml are kept."""
        p = Path(path)
        if not (p / "config.yaml").is_file():
            raise FileNotFoundError(f"{p} is not a run folder (no config.yaml)")
        return cls(config, profile, reopen=p)

    @property
    def eval_dir(self) -> Path:
        d = self.path / "eval"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def resume_path(self) -> Path:
        return self.checkpoints / RESUME_NAME

    def save_teacher(
        self,
        state: dict[str, Any],
        spec: ActionSpec,
        *,
        it: int | None = None,
        obs_dim: int,
        sidecar: dict[str, Any],
    ) -> Path:
        """``teacher.pt`` (or ``teacher_it<N>.pt``) + its sidecar, atomically."""
        import torch

        if spec.version == CONTRACT_VERSION and spec.max_g is None:
            spec = replace(spec, max_g=float(self.profile.max_g))
        name = TEACHER_NAME if it is None else f"teacher_it{it}.pt"
        path = self.checkpoints / name
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
        write_sidecar(
            path, spec, extra_dim=obs_dim, actor=self.config.actor.kind, drone=self.profile.id,
            run_id=self.run_id, **sidecar,
        )  # fmt: skip
        return path

    def save_resume(self, state: dict[str, Any]) -> Path:
        import torch

        tmp = self.resume_path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.resume_path)
        return self.resume_path

    def progress(self, row: dict[str, Any]) -> None:
        path = self.logs / "progress.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def write_tracks_snapshot(self, pool: Any) -> Path:
        """``tracks/``: the catalogues, ``courses.json`` and ``files.sha256`` (train files only)."""
        d = self.path / "tracks"
        d.mkdir(parents=True, exist_ok=True)
        for name, cat in pool.catalogues.items():
            (d / f"catalogue-{name}.yaml").write_text(
                yaml.safe_dump(cat.as_dict(), sort_keys=False, width=100), encoding="utf-8"
            )
        (d / "courses.json").write_text(json.dumps(pool.table(), indent=1) + "\n", "utf-8")
        lines = []
        for path, info in sorted(pool.record.files.items()):
            extra = " ".join(f"{k}={v}" for k, v in info.items() if k != "sha256")
            lines.append(f"{info.get('sha256')}  {path}  {extra}")
        (d / "files.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return d

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
            obs_contract_version=obs_contract_version(extra_dim),
            extra=_slew_sidecar_extra(self.config),
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
