"""UC69 AC7: the oracle gate source can never feed training or fine-tuning.

The guard is fail-closed (an unstated build purpose is refused), no training config accepts an
oracle key, no training CLI exposes it, and no training module imports it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from drone_fly.harness.errors import OracleLeakageError
from drone_fly.harness.interfaces import BuildContext
from drone_fly.harness.oracle import (
    ORACLE_ALLOWED_PURPOSES,
    TRAINING_PURPOSES,
    OracleGateSource,
    require_oracle_purpose,
)
from drone_fly.training.config import TrainConfigError, load_teacher_config, load_train_config
from tests.oracle_support import make_course

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("purpose", ["eval", "diagnostic"])
def test_eval_and_diagnostic_are_allowed(purpose):
    require_oracle_purpose(purpose)
    OracleGateSource(make_course(), purpose=purpose)


@pytest.mark.parametrize("purpose", [None, "", "train", "fine-tune", "finetune", "Eval", "student"])
def test_everything_else_is_refused(purpose):
    with pytest.raises(OracleLeakageError, match="diagnostic"):
        require_oracle_purpose(purpose)
    with pytest.raises(OracleLeakageError):
        OracleGateSource(make_course(), purpose=purpose)


def test_training_purposes_are_outside_the_allow_list():
    assert not (TRAINING_PURPOSES & ORACLE_ALLOWED_PURPOSES)
    assert ORACLE_ALLOWED_PURPOSES == {"eval", "diagnostic"}


def test_default_build_context_refuses():
    ctx = BuildContext(config=None, contract=None, extra_dim=148)
    assert ctx.purpose is None
    ctx.resources["oracle_course"] = make_course()
    with pytest.raises(OracleLeakageError):
        OracleGateSource.from_config({}, ctx)
    ctx.purpose = "train"
    with pytest.raises(OracleLeakageError):
        OracleGateSource.from_config({}, ctx)


def test_registry_built_oracle_refuses_without_purpose():
    """Building the registered ``oracle`` part through the registry is guarded too."""
    from drone_fly.harness.config import load_config
    from drone_fly.harness.registry import build_part

    cfg = load_config("liftoff")
    ctx = BuildContext(config=cfg, contract=None, extra_dim=148)
    ctx.resources["oracle_course"] = make_course()
    with pytest.raises(OracleLeakageError):
        build_part("gate_estimator", "oracle", {}, ctx)
    ctx.purpose = "eval"
    assert isinstance(build_part("gate_estimator", "oracle", {}, ctx), OracleGateSource)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"perception": "oracle"}, "unknown training config keys"),
        ({"gate_estimator": "oracle"}, "unknown training config keys"),
        ({"tracks": {"perception": "oracle"}}, "unknown tracks keys"),
        ({"tracks": {"gate_estimator": "oracle"}}, "unknown tracks keys"),
    ],
)
def test_teacher_training_config_rejects_oracle_keys(override, match):
    with pytest.raises(TrainConfigError, match=match):
        load_teacher_config("teacher-liftoff", override, validate_tracks=False)


@pytest.mark.parametrize("override", [{"perception": "oracle"}, {"gate_estimator": "oracle"}])
def test_legacy_training_config_rejects_oracle_keys(override):
    with pytest.raises(TrainConfigError, match="unknown training config keys"):
        load_train_config("proc-course-dr", override)


def _walk(parser: argparse.ArgumentParser, path: str = ""):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield from _walk(sub, f"{path} {name}".strip())
        else:
            yield path, action


def test_no_training_cli_exposes_the_oracle():
    from drone_fly.cli.main import build_parser

    seen_train = False
    for path, action in _walk(build_parser()):
        if not path.startswith("train"):
            continue
        seen_train = True
        names = " ".join(action.option_strings + [str(action.dest)]).lower()
        choices = " ".join(map(str, action.choices or [])).lower()
        default = str(action.default).lower()
        for text in (names, choices, default):
            assert "oracle" not in text and "perception" not in text, (path, action)
    assert seen_train
    # ...while the harness run does expose it (sanity of the walk)
    flags = [a.option_strings for p, a in _walk(build_parser()) if p == "harness run"]
    assert ["--perception"] in flags


def test_training_modules_never_import_the_oracle():
    code = (
        "import sys\n"
        "import drone_fly.training.cli, drone_fly.training.ppo, drone_fly.training.config, "
        "drone_fly.training.evaluate, drone_fly.training.run_dir\n"
        "bad = [m for m in ('drone_fly.harness.oracle', 'drone_fly.harness.teacher', "
        "'drone_fly.harness.scoring') if m in sys.modules]\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO / "src") + os.pathsep + str(REPO)}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                       cwd=str(REPO), timeout=300)  # fmt: skip
    assert p.returncode == 0, (p.stdout, p.stderr)
