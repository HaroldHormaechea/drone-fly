# Flying a drone with a real fly brain — the connectome-as-reservoir approach

This document explains, from the ground up, how the project trains the **real K1 connectome** (a
25,627-neuron / ~3.86M-synapse slice of the MaleCNS fly brain) to fly a quadcopter through a gate
course — and the key engineering idea, *the connectome as a frozen reservoir*, that made it both
fast and biologically honest. It is written to be understandable without a background in
reinforcement learning, control theory, or neuroscience; software analogies are used throughout.

---

## The three ingredients

Every result here is a combination of a **brain**, a **task**, and a **learning rule**.

### 1. The brain — the K1 connectome

A *connectome* is a literal wiring diagram of a real nervous system: which neuron connects to which,
and how strongly. Think of it as a **directed graph**:

- ~25,627 nodes (neurons),
- ~3.86 million edges (synapses), each carrying a **weight** (strength) and a **sign** (excitatory `+`
  or inhibitory `−`).

These numbers are not invented — they are measured from a real fly-brain dataset (MaleCNS). "K1" is
the specific pruned slice this project uses (`artifacts/pruned/k1/`). When we say "the actual brain,"
we mean the 3.86M-edge graph loaded with its *real measured synaptic weights*, with signals run
through it.

### 2. The task — flying the gate course

A GPU-parallel physics simulator flies a quadcopter that must pass through a sequence of gates. Each
instant, twelve numbers describe the drone's situation (its position/velocity/orientation relative to
the next gate) — the **observation**. The drone outputs four numbers — throttle plus three rotation
rates — the **action**. Fifty times a second: observe → decide → move.

The control interface is **CTBR / "acro" mode** (collective thrust + body rates, *no* auto-leveling).
This is deliberately the same low-level interface real FPV quadcopters and sim racers (e.g. Liftoff)
use, which is what makes later transfer plausible.

For speed, **thousands of drones are simulated in parallel on the GPU** as one batched tensor — the
source of the ~100–1000× throughput over the previous single-drone (pybullet) simulator.

### 3. The learning rule — reinforcement learning (PPO)

Nobody tells the drone the correct action. Instead it earns a **reward** (large bonus for passing a
gate / finishing, small bonus for forward progress and staying upright, penalty for crashing). PPO is
**guided trial-and-error**: try actions, statistically keep the nudges that led to more reward. If you
have done security fuzzing, it is like fuzzing toward a goal where a reward gradient steers the search
instead of random mutation.

---

## The key idea: the brain as a *frozen reservoir*

This is the move that made training the full K1 practical.

**The problem.** If you let the learning algorithm adjust all 3.86M synapse weights (i.e. *train the
brain itself*), every learning step must push error backwards through all 3.86 million edges, dozens
of times per cycle. Measured on this hardware: **~368 steps/second → ~19 hours** for a full run. Too
slow to iterate on.

**The insight — "reservoir computing."** You do not have to retrain the brain. You can treat the fixed
brain as a **feature-generating black box** and train only a thin "readout" that listens to it.

> **Software analogy.** It is like depending on a large pretrained **vendor binary** you never
> recompile. Recompiling it on every build (training all 3.86M weights) is brutally slow. Instead you
> leave the binary fixed and write a **small adapter** that reads its outputs and makes decisions. You
> iterate on the ~150-line adapter, not the giant library.

Concretely:

- **Frozen (never changes):** the entire K1 brain — its wiring *and* its real synaptic weights. Every
  observation is injected into a small set of sensory neurons and propagates through the real
  connectome dynamics (`tanh(W · state)`, several steps), producing a 25,627-dimensional internal
  "brain state."
- **Trained (tiny, ~150k parameters):** a readout that **taps 1,024 of those neurons** and maps them
  to the four control outputs (a LayerNorm, then a small MLP).

**Why this is a *better* scientific story, not a shortcut.** In a real fly you do not rewrite the
connectome by gradient descent either — the wiring is fixed anatomy, and learning happens in how
downstream circuits *read* it. Freezing the brain and training the readout is the more biologically
honest setup. And it is ~11× faster: **~4,100 steps/second, ~1.7 hours** for a full run (in practice
the task is solved in ~24 minutes).

**The speed trick that makes freezing pay off twice.** Because the brain is fixed, its output for a
given observation never changes — so each observation's brain-features are computed **once** during
data collection and **reused** across the algorithm's repeated optimization passes. The expensive
brain forward never runs inside the learning update at all.

---

## Evidence that the brain is doing the work (not decorative)

A fair question: if only the readout is trained, is the frozen brain actually contributing, or could
any random transform do?

The first attempt tapped only **256** neurons and **stalled at ~1% completion** — the drone flew past
the first gate then crash-dived into the second at a steep angle. The fix was to **tap 1,024 neurons
instead of 256** (and normalize them). With the *same frozen brain*, completion **jumped from ~1% to
~59%**, flying upright and controlled.

The inference: the same brain, merely *read from more widely*, went from failing to flying. If the
brain were decorative and the readout did everything, tapping more of its neurons would not matter.
Tapping more = accessing more of the brain's internal computation = better flight. A second
observation from the recordings reinforces this: only ~50 of the 25,627 neurons carry the flight
signal meaningfully — the computation is **sparse**, concentrated in a specific subpopulation, exactly
what you would expect if real circuit structure is being exploited.

---

## Reading the numbers

| Metric | Meaning |
|---|---|
| **completion** | % of the parallel drones that fly through all gates to the finish. The honest headline is the *deterministic* eval (exploration noise off): **100%**. |
| **fps** | simulated control-steps per second (throughput). ~4,100 for the frozen-K1 reservoir. |
| **tilt** | how far from level the drone is. ~7° mean at convergence = controlled banking, not a tumble. |

> **Why tilt is tracked separately.** A drone can "complete" a course while ballistically tumbling
> (spinning through the gates by luck). Tracking tilt proves the flight is genuinely *controlled and
> upright*, not a lucky tumble — an early version of this project was briefly fooled by exactly that.

---

## The result

The real 25,627-neuron / 3.86M-synapse K1 connectome, run **frozen** (real biological weights, never
gradient-edited), with only a ~150k-parameter readout learning to listen to 1,024 of its neurons,
flies the gate course at **100% deterministic completion, ~7° mean tilt** — stable, upright,
controlled flight — trained in ~24 minutes on a single 6 GB GPU.

Recordings of the brain in flight (per-frame activation of all 25,627 neurons, with anatomical neuron
metadata) are committed under `training/k1-gate-flight/recordings/` and viewable in the desktop app.

### Implementation pointers

- `gpu_prototype/gates_k1_reservoir.py` — the frozen-K1 reservoir policy + PPO training loop with
  feature caching.
- `gpu_prototype/gates_eval_res.py` — deterministic evaluation (completion + tilt profile).
- `gpu_prototype/record_k1_gates.py` — exports app-readable recordings via the repo `ActivationRecorder`.
- `gpu_prototype/bench_k1.py` — the throughput benchmark behind the 368 vs 4,100 fps figures.
