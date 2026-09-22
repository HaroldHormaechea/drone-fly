"""Best-effort training-time activation recording callback (UC-05, AC4).

Recording during **evaluation** (see :mod:`drone_fly.evaluate.evaluator`) is the tested,
deterministic primary path. This callback adds the *documented best-effort* training path:
it captures env-0's every-Nth episode during PPO's on-policy rollout collection, so you can
watch the brain change as it learns. It is intentionally defensive — any capture error
disables recording and logs, but never crashes training — because training is
mid-optimisation with sampled (non-deterministic) actions and normalised rewards, so its
playbacks are illustrative rather than reproducible.

Only env 0 is recorded. The connectome actor's ``sink`` (wired at training start) stashes
the batched ``(n_envs, N)`` post-propagation state each forward; on each ``_on_step`` we pull
env-0's row and pair it with env-0's **applied** action / ``info["position"]``. Episode
boundaries are detected from ``dones[0]``; the recorded ``total_reward`` is the sum of the
*normalised* per-step rewards seen during training (documented caveat).

Action fidelity (UC-45 AC3)
---------------------------
The recorded action is exactly the array handed to ``env.step`` — SB3's ``clipped_actions``
(the raw Gaussian sample after clip / unscale-into-box), NOT the unclipped ``actions``
sample. This matches the evaluation recorder, which already records the actor's stepped
output, so a recorded trajectory always replays to the true drone state. (Falls back to
``actions`` with a one-time warning only if ``clipped_actions`` is absent from ``locals``.)
"""

from __future__ import annotations

import glob
import logging
import os
import re

from stable_baselines3.common.callbacks import BaseCallback

from drone_fly.controller.sb3 import actor_from_model
from drone_fly.record.recorder import ActivationRecorder

logger = logging.getLogger(__name__)

_EPISODE_RE = re.compile(r"episode_(\d+)\.json(\.gz)?$")


def _highest_episode_index(record_dir) -> int:
    """Return the highest ``n`` across ``episode_<n>.json[.gz]`` files in ``record_dir``.

    Scans both plain ``.json`` and gzipped ``.json.gz`` playback files. Returns ``-1`` for a
    missing or empty directory, so callers can add 1 to get the next index to write (empty →
    0). This is how a resumed run continues past the prior run's recordings instead of
    clobbering them.
    """
    highest = -1
    for suffix in ("episode_*.json", "episode_*.json.gz"):
        for path in glob.glob(os.path.join(str(record_dir), suffix)):
            m = _EPISODE_RE.search(os.path.basename(path))
            if m:
                highest = max(highest, int(m.group(1)))
    return highest


class RecordingCallback(BaseCallback):
    """Record env-0's every-``record_every``-th training episode into ``recorder``."""

    def __init__(
        self,
        recorder: ActivationRecorder,
        *,
        record_every: int = 1,
        seed: int = 0,
        tw_preserving: bool = True,
    ):
        super().__init__()
        self.recorder = recorder
        self.record_every = max(int(record_every), 1)
        self.seed = int(seed)
        # UC-49 AC3: the run's ``EnvConfig.pybullet_tw_preserving`` flag, forwarded to
        # :func:`~drone_fly.adapter.dynamics_summary.drone_dynamics_summary` so the recorded
        # ``meta.drone_dynamics`` pins the same UC-48-resolved applied mass / T/W the env uses.
        self._tw_preserving = bool(tw_preserving)
        self._actor = None
        self._episode = 0
        self._capturing = False
        self._total_reward = 0.0
        self._steps = 0
        self._enabled = True
        # UC-45: emit the "clipped_actions absent" fallback warning at most once per run.
        self._warned_no_clipped = False

    def _begin_episode(self) -> None:
        self._capturing = (self._episode % self.record_every) == 0
        self._total_reward = 0.0
        self._steps = 0
        if self._capturing:
            self.recorder.start_episode(self._episode, self.seed)
            # UC-08 AC9 (best-effort): stamp env-0's per-episode active course so a
            # randomized-training playback shows the true sampled geometry. Guarded — an
            # env without the property must never crash the training callback.
            try:
                self.recorder.set_course(self.training_env.get_attr("active_course")[0])
            except Exception:  # noqa: BLE001 - best-effort; recording never breaks training
                pass
            # UC-45 AC9 (best-effort): stamp env-0's per-episode sampled dynamics so a
            # randomized-training playback pins the true sampled physics. Guarded exactly like
            # set_course — an env without the property (or randomization off → None) must never
            # crash the training callback; a None stamp simply omits the meta.dynamics block.
            try:
                self.recorder.set_dynamics(self.training_env.get_attr("active_dynamics")[0])
            except Exception:  # noqa: BLE001 - best-effort; recording never breaks training
                pass
            # UC-49 AC3 (best-effort): stamp env-0's drone-dynamics summary so a randomized-
            # training playback pins the resolved physics (applied mass / weight / T/W / hover /
            # curriculum knobs) via the SAME shared computation the TUI and CI guard use. Its own
            # guarded block — an env without the accessors (or randomization off → default
            # DynamicsParams) must never crash the training callback. Uses the run's tw_preserving
            # flag so the recorded T/W matches the env's UC-48 resolution.
            try:
                from drone_fly.adapter.dynamics_summary import drone_dynamics_summary
                from drone_fly.env.config import DynamicsParams

                backend = self.training_env.get_attr("backend")[0]
                dyn = self.training_env.get_attr("active_dynamics")[0] or DynamicsParams()
                try:
                    attitude_authority = float(self.training_env.get_attr("attitude_authority")[0])
                except Exception:  # noqa: BLE001 - env may predate the accessor; default full
                    attitude_authority = 1.0
                try:
                    spawn_z = self.training_env.get_attr("spawn_z")[0]
                except Exception:  # noqa: BLE001 - env may predate the accessor; leave unknown
                    spawn_z = None
                self.recorder.set_drone_dynamics(
                    drone_dynamics_summary(
                        backend=backend,
                        sampled_mass=dyn.mass,
                        max_body_rate=dyn.max_body_rate,
                        max_thrust=dyn.max_thrust,
                        tw_preserving=self._tw_preserving,
                        attitude_authority=attitude_authority,
                        spawn_z=spawn_z,
                    )
                )
            except Exception:  # noqa: BLE001 - best-effort; recording never breaks training
                pass
            if self._actor is not None:
                self._actor.sink = self.recorder.sink
        elif self._actor is not None:
            self._actor.sink = None

    def _on_training_start(self) -> None:
        try:
            self._actor = actor_from_model(self.model)
        except TypeError as exc:  # non-connectome policy — nothing to record
            logger.warning("Recording disabled: %s", exc)
            self._enabled = False
            return
        # Continue numbering past any existing recordings in this dir so a resumed run never
        # clobbers the prior run's episode_<n>.json[.gz] (empty dir → 0, fresh-run parity).
        # Note: this counter is derived from the files on disk, so with record_every>1 it
        # will NOT equal the prior run's cumulative episode count — that is intentional (it
        # guarantees no clobber while preserving the absolute-index cadence); don't "correct" it.
        self._episode = _highest_episode_index(self.recorder.out_dir) + 1
        self._begin_episode()

    def _on_step(self) -> bool:
        if not self._enabled:
            return True
        try:
            infos = self.locals.get("infos") or []
            dones = self.locals.get("dones")
            actions = self.locals.get("actions")
            rewards = self.locals.get("rewards")
            if not infos or dones is None or actions is None:
                return True
            info0 = infos[0]
            if self._capturing and "position" in info0:
                # UC-45 AC3 (action-channel fidelity): record the action actually APPLIED to the
                # env, not the raw PPO Gaussian sample. SB3's on-policy collector clips (or, under
                # ``squash_output``, unscales) ``actions`` into the action space and steps the env
                # with the result, exposing it as ``clipped_actions`` in ``locals``. Recording
                # ``actions[0]`` logged the unclipped sample (throttle/roll could sit outside the
                # box) — faithful today only by accident (``sanitize_action`` == ``np.clip`` with
                # ``squash_output=False``) and silently desyncing under a squashed head or changed
                # bounds. Prefer ``clipped_actions[0]``; fall back to ``actions[0]`` (with a one-
                # time warning) only if it is absent, so the recorder stores exactly the array
                # handed to ``env.step``.
                clipped = self.locals.get("clipped_actions")
                if clipped is not None:
                    applied_action = clipped[0]
                else:
                    if not self._warned_no_clipped:
                        logger.warning(
                            "clipped_actions absent from callback locals; recording the raw "
                            "actions sample (may lie outside the action box)."
                        )
                        self._warned_no_clipped = True
                    applied_action = actions[0]
                self.recorder.capture_frame(
                    applied_action, info0["position"], target_gate=info0.get("target_gate")
                )
                self._steps += 1
                if rewards is not None:
                    self._total_reward += float(rewards[0])
            if bool(dones[0]):
                if self._capturing:
                    self.recorder.finish_episode(
                        completed=bool(info0.get("completed", False)),
                        completion_time=info0.get("completion_time"),
                        total_reward=self._total_reward,
                        steps=self._steps,
                    )
                self._episode += 1
                self._begin_episode()
        except Exception as exc:  # never crash training over a recording glitch
            logger.warning("Disabling training-time recording after error: %s", exc)
            self._enabled = False
            if self._actor is not None:
                self._actor.sink = None
        return True

    def _on_training_end(self) -> None:
        if self._actor is not None:
            self._actor.sink = None
