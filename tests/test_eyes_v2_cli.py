"""UC-83 AC7: the new ``eyes`` CLI surface -- v2 train/dataset flags and the ``eyes compare``
verdict subcommand (parser shapes + a hermetic end-to-end ``compare`` through ``main``).
"""

from __future__ import annotations

import json

import pytest

from drone_fly.cli.main import EXIT_OK, build_parser, main


def _p(*argv):
    return build_parser().parse_args(["eyes", *argv])


def test_train_v2_flag_defaults_are_off():
    a = _p("train", "--dataset", "d")
    assert a.hit_head is False and a.focal is False and a.hit_sampler is False
    assert (a.focal_alpha, a.focal_gamma, a.w_focal, a.hit_sample_w) == (0.25, 2.0, 1.0, 8.0)


def test_train_v2_flags_parse():
    a = _p(
        "train", "--dataset", "d", "--hit-head", "--focal", "--hit-sampler",
        "--focal-alpha", "0.3", "--focal-gamma", "1.5", "--w-focal", "2.0", "--hit-sample-w", "6",
    )
    assert a.hit_head is True and a.focal is True and a.hit_sampler is True
    assert (a.focal_alpha, a.focal_gamma, a.w_focal, a.hit_sample_w) == (0.3, 1.5, 2.0, 6.0)


def test_dataset_mining_flags():
    a = _p("dataset")
    assert a.hit_frac is None and a.mining_seed == 0
    a = _p("dataset", "--hit-frac", "0.4", "--mining-seed", "5")
    assert a.hit_frac == 0.4 and a.mining_seed == 5


def test_compare_parser_defaults_and_required():
    a = _p("compare", "--v1", "a.json", "--v2", "b.json")
    assert (a.v1, a.v2) == ("a.json", "b.json")
    assert a.min_rel_reduction == 0.30 and a.max_fa_clear == 0.10
    for argv in (["compare", "--v1", "a.json"], ["compare", "--v2", "b.json"]):
        with pytest.raises(SystemExit):
            _p(*argv)


def _report(nearest_quarter_miss, fa_clear_all):
    return {
        "distance_bins": [{"bin": "0-25%", "in_view": {"miss_rate": nearest_quarter_miss}}],
        "aggregate": {
            "all": {"false_alarm_rate_clear": fa_clear_all},
            "in_view": {"miss_rate": 0.5},
        },
    }


def test_compare_end_to_end_via_main(tmp_path, capsys):
    (tmp_path / "v1.json").write_text(json.dumps(_report(1.0, 0.03)))
    (tmp_path / "v2.json").write_text(json.dumps(_report(0.6, 0.05)))
    out = tmp_path / "cmp"
    rc = main([
        "eyes", "compare",
        "--v1", str(tmp_path / "v1.json"),
        "--v2", str(tmp_path / "v2.json"),
        "--out", str(out),
    ])
    assert rc == EXIT_OK
    printed = json.loads(capsys.readouterr().out.split("compare:")[0])
    assert printed["passed"] is True and printed["verdict"].startswith("PASS")
    assert (out / "compare.json").is_file() and (out / "compare.md").is_file()
