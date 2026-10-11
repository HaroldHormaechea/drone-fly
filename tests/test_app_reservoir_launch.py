"""UC-63: the generic ``train_aug`` launcher's pure argument resolver (no torch, hermetic).

Only :func:`app.reservoir_launch.parse_args` / :func:`resolve_kwargs` are exercised; ``main()``
imports ``gpu_prototype`` (torch) and is never called here.
"""

from __future__ import annotations

import pytest
from app import configs_io, reservoir_launch
from app.reservoir_launch import parse_args, resolve_kwargs


def test_only_save_path_when_no_knob_is_set() -> None:
    assert resolve_kwargs(parse_args(["--save-path", "/r/m.pt"]), 70) == {"save_path": "/r/m.pt"}


def test_set_knobs_are_forwarded_with_their_types() -> None:
    args = parse_args(
        [
            "--save-path",
            "m.pt",
            "--N",
            "256",
            "--budget",
            "3e6",
            "--roll",
            "16",
            "--epochs",
            "8",
            "--logstd-init",
            "-0.7",
            "--logstd-final",
            "-1.5",
            "--anneal-start",
            "0.3",
            "--count-curr-frac",
            "0.5",
            "--pos-curr-frac",
            "0.25",
        ]
    )
    kwargs = resolve_kwargs(args, 70)
    assert kwargs == {
        "save_path": "m.pt",
        "N": 256,
        "budget": 3e6,
        "roll": 16,
        "epochs": 8,
        "logstd_init": -0.7,
        "logstd_final": -1.5,
        "anneal_start": 0.3,
        "count_curr_frac": 0.5,
        "pos_curr_frac": 0.25,
    }
    assert isinstance(kwargs["N"], int) and isinstance(kwargs["budget"], float)
    assert "warm_from" not in kwargs and "warm_src_extra_dim" not in kwargs


def test_warm_src_extra_dim_defaults_to_runtime_extra_dim_when_warm_from_is_set() -> None:
    kwargs = resolve_kwargs(parse_args(["--save-path", "m.pt", "--warm-from", "/w.pt"]), 70)
    assert kwargs == {"save_path": "m.pt", "warm_from": "/w.pt", "warm_src_extra_dim": 70}


def test_explicit_warm_src_extra_dim_is_kept() -> None:
    args = parse_args(["--save-path", "m.pt", "--warm-from", "/w.pt", "--warm-src-extra-dim", "30"])
    assert resolve_kwargs(args, 70)["warm_src_extra_dim"] == 30


def test_warm_src_extra_dim_is_dropped_without_warm_from() -> None:
    args = parse_args(["--save-path", "m.pt", "--warm-src-extra-dim", "30"])
    assert resolve_kwargs(args, 70) == {"save_path": "m.pt"}


def test_save_path_is_required() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--N", "4"])


def test_bad_type_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--save-path", "m.pt", "--N", "1.5"])


def test_every_schema_knob_has_an_option() -> None:
    parser = reservoir_launch.build_parser()
    options = {opt for action in parser._actions for opt in action.option_strings}
    for knob in configs_io.KNOB_NAMES:
        assert "--" + knob.replace("_", "-") in options


def test_module_level_has_no_gpu_prototype_imports() -> None:
    # gpu_prototype is imported inside main() only (fresh-interpreter check: test_app_imports).
    for name in ("torch", "train_aug", "BatchedProcCourse", "EXTRA_DIM"):
        assert not hasattr(reservoir_launch, name)
