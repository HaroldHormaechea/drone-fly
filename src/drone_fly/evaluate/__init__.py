"""Evaluation stage: run a trained agent and report performance (UC-03).

Loads a checkpoint + its VecNormalize stats, flies N deterministic episodes on the racing
course, and reports the course-completion rate and mean start→gate→finish time with pinned
semantics (see :mod:`drone_fly.evaluate.evaluator`).
"""

from __future__ import annotations

from drone_fly.evaluate.evaluator import EvalMetrics, evaluate_checkpoint

__all__ = ["EvalMetrics", "evaluate_checkpoint"]
