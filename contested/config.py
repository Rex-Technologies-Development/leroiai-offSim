"""Strict configuration schema and loader for the canonical environment.

Follows the ``offsim/sim/config.py`` spirit (single YAML contract) but adds a
strict loader that *rejects unknown keys*, as required by plan Section 3.1. The
knob is ``alpha``; ``tau_rev`` is derived, and ``alpha == 0`` maps to
``tau_rev = inf`` through an explicit branch (never a large finite number).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"

_CONTEST_MODES = ("majority", "suppress", "none")
_WEIGHT_DISTS = ("uniform", "lognormal", "bimodal")
_LAYOUTS = ("uniform", "clustered", "polarized")


@dataclass(frozen=True)
class CanonicalConfig:
    """Immutable, validated spec for one canonical-environment configuration.

    ``alpha`` is the sweep knob; ``tau_rev`` is derived via the :pyattr:`tau_rev`
    property. All node counts must be <= their ``max_*`` padding bound so a single
    set of model weights runs across configurations.

    Not an arbitrary abstract world: ``field_size`` defaults to 3.66 m (144 in) and
    ``horizon_T`` to 120 s, *deliberately matching Override*. The canonical env is
    Override's geometry and clock with Override's scoring stripped out — which is
    also why results here are comparable to the Override game (plan Section 1.1).
    """

    # episode
    dt: float = 0.05
    decision_dt: float = 0.5
    horizon_T: float = 120.0

    # counts
    n_robots: int = 4
    n_tasks: int = 12
    n_adversaries: int = 4
    max_robots: int = 8
    max_tasks: int = 20
    max_adversaries: int = 8

    # field
    field_size: tuple[float, float] = (3.66, 3.66)
    service_radius: float = 0.20

    # dynamics
    v_max: float = 1.5
    a_max: float = 3.0
    adv_v_max: float = 1.5
    adv_a_max: float = 3.0

    # the contested mechanic
    tau_com: float = 2.0
    alpha: float = 1.0
    beta: float = 1.0
    contest_mode: str = "majority"

    # re-acquisition cost rho (consultant): a task completed BEFORE needs tau_com * rho of
    # service to complete AGAIN (the first completion always costs tau_com). rho=1 (default)
    # = re-completion as cheap as first completion = the regime where retention provably
    # CANNOT pay (perfect retention knowledge ties weight-only). Override has rho >> 1 —
    # re-scoring a descored goal is a full fetch-and-place. The payoff to retention-aware
    # allocation scales with rho: this is the swept parameter that turns the ties into wins.
    reacquire_cost: float = 1.0

    # task layout
    weight_dist: str = "lognormal"
    weight_range: tuple[float, float] = (1.0, 5.0)
    layout: str = "clustered"
    n_clusters: int = 3

    # structural retention variance (analogue of Override's protected Alliance Goals):
    # a fraction of tasks are PROTECTED — once completed they can never be reverted
    # (per-task tau_rev = inf), giving retention that varies by task structure, not
    # just by transient adversary position. 0.0 recovers the all-reversible world.
    protected_fraction: float = 0.0

    # killer-control knob (consultant Claim-4 test): when True, the observation exposes
    # each task's PROTECTED flag as an extra task feature. A weight-only ("off") policy
    # with this on is handed "which goals are protected" for free — the control that
    # distinguishes a learned retention estimate from a mere relabelling of protection.
    expose_protected: bool = False

    # M8 toggle observability: when True (and toggle_regions>0), the obs exposes the toggle leverage
    # (is_toggle, which team holds each task's governing toggle) and the decoder gates the toggle-
    # EFFECTIVE value instead of base weight — so a retention-aware policy can SEE and defend the
    # multiplier control points. False = retention-blind (base weight only): the M8 ablation.
    expose_toggle: bool = False

    # ablation (paper Table III row): when True the observation hides ALL adversary nodes —
    # adv_feat and the e_at adversary->task edges are zeroed and adv_valid is forced False, so
    # the encoder and retention head cannot see adversary geometry. Adversaries still ACT in the
    # env (they still revert tasks); the policy is just blind to them. Tests whether R_hat's
    # discrimination and the allocation come from reading adversary positions. Obs shapes are
    # unchanged (nodes are zeroed, not removed), so one set of weights still runs everywhere.
    ablate_adversary_nodes: bool = False

    # PHASE 2 master switch (consultant round 5 — the symmetry fix). When False (default)
    # the env is exactly as before: blue acquires/defends, red only reverts, byte-identical
    # dynamics and reference match. When True the game becomes SYMMETRIC — red also completes
    # and holds tasks (three-valued ownership neutral/blue/red), and blue gains a REVERSE
    # action to flip red-held tasks (→ neutral, re-acquire at tau_com*rho). This is what makes
    # estimating retention on the OPPONENT's tasks a real, graded, geometric prediction problem
    # — the decision current allocators cannot represent. See tenure-phase2-plan.md.
    # NOTE: the symmetric tick/action/obs machinery lands incrementally (P2-A..D); until then
    # this flag only gates config-level scaffolding and is a no-op for the dynamics.
    symmetric: bool = False

    # PHASE 3 (consultant/user — retention as something you BUILD, not just predict). When
    # stack_cap > 1 the symmetric env becomes a STACKING game: a task has an integer height
    # (h in [0, stack_cap]). Placing the piece at level L costs tau_com * L and DISMANTLING the piece
    # at level L costs tau_rev * L -- reaching higher is slower on BOTH sides (the arm must travel to
    # level L to add OR remove a piece). So building a full stack costs tau_com * h(h+1)/2 and
    # flattening it costs tau_rev * h(h+1)/2 -- both quadratic. A tall stack is a big up-front
    # investment (slow to raise) that is then correspondingly expensive to tear down, so retention
    # becomes a thing the allocator PRODUCES (concentrate on defensible tasks) rather than merely
    # estimates. stack_cap = 1 (default) recovers the non-stacking take model exactly (level-1 build
    # costs tau_com, byte-identical, so Phase 1/2 are unaffected). No-op unless cfg.symmetric.
    stack_cap: int = 1

    # NON-LINEAR stack value (user — retention on stacks, not just toggles). Held value of a height-h
    # stack is w * h ** stack_value_power. power = 1.0 (default) is the linear "objects are
    # interchangeable" world where a height-h stack == h height-1 tasks and retention is provably
    # DECORATIVE. power > 1 makes value CONVEX: a tall stack is worth disproportionately more than the
    # same pieces spread flat, and the MARGINAL piece on a height-h stack ((h+1)^p - h^p) is worth more
    # than starting a new one -- so a tall stack is a defensible control point whose loss is outsized,
    # exactly the condition under which retention PAYS (the same mechanism as the toggle multiplier,
    # applied to height). power < 1 is concave (diminishing). No-op at stack_cap == 1 (h in {0,1}).
    stack_value_power: float = 1.0

    # PHASE 3 TOGGLES (Override-grounded — retention as MULTIPLIER control). Real Override retention
    # is not equal-value stacking: owning a quadrant's TOGGLE turns its goals from 5 to 15 points
    # (a 3x on a whole cluster). Modelled here as ``toggle_regions`` regions, the first task of each
    # being its TOGGLE; while a team owns a region's toggle, that team's held value from the region's
    # GOALS is multiplied by ``toggle_multiplier``. This makes a few scarce points HIGH-LEVERAGE:
    # holding one multiplies a cluster, losing it collapses the cluster to 1x -- so retention finally
    # PAYS (a retention-aware allocator that prices+defends toggles beats a blind one iff M>1 and a
    # smart opponent contests them; decorative at M=1, verified by experiments/toggle_retention.py).
    # toggle_multiplier = 1.0 (default) is an exact no-op. Requires cfg.symmetric and toggle_regions>0.
    toggle_multiplier: float = 1.0
    toggle_regions: int = 0

    # DYNAMIC cluster exposure (consultant). A toggle's value is not a static premium but the value it
    # CURRENTLY unlocks: base + (M-1) * (held value of its cluster's goals RIGHT NOW). Early (empty
    # cluster) a toggle governs nothing and is nearly worthless; late (loaded cluster) losing it is
    # catastrophic -- the same toggle's importance changes minute to minute with the board. This turns
    # the toggle STEP-function into a CONTINUOUS, state-dependent value SPECTRUM that blind cannot read
    # off two fixed positions, and gives the retention head its first graded (non-constant, non-flag)
    # target. When True (and toggle_regions>0): the obs value channel uses current exposure AND cluster
    # sizes are VARIED (wider exposure range). Default False = byte-identical. Surface it with
    # expose_toggle. See dynamic-exposure-design.md.
    dynamic_exposure: bool = False

    # retention label definition. Default R integrates the held fraction over the WHOLE
    # remaining episode (includes the policy's own later re-completions), which over-values
    # tasks under churn. With this True, R is the MARGINAL "no-revisit counterfactual":
    # held only until the FIRST reversal after completion (assuming no re-grab). See
    # contested/labels.py. Fixes the double-count the consultant flagged for α≈1.5–2.
    retention_no_revisit: bool = False

    # allocation
    allow_multi_defend: bool = True
    allow_multi_acquire: bool = False
    # PHASE 2 (consultant): two robots may commit to the SAME reverse target, so a defended red
    # task (a red robot sitting on it) can be out-numbered and taken. Without this a reverse is a
    # 1v1 standoff that never flips and R-take has no signal. No-op outside symmetric mode.
    allow_multi_reverse: bool = True

    # adversaries
    adversary_population: tuple[str, ...] = ("greedy_nearest", "value_targeting", "feinter")
    holdout_population: tuple[str, ...] = ("camper", "learned_selfplay")

    seed: int = 0

    # ------------------------------------------------------------------ derived
    @property
    def tau_rev(self) -> float:
        """Reversal dwell, derived from ``alpha``.

        ``alpha == 0`` means "no reversal is ever fast enough" -> ``tau_rev = inf``.
        Handled with an explicit branch so ``eta >= tau_rev`` is simply never true,
        with no division by zero and no magic large finite constant.
        """
        if self.alpha <= 0.0:
            return math.inf
        return self.tau_com / self.alpha

    @property
    def ticks_per_decision(self) -> int:
        return int(round(self.decision_dt / self.dt))

    @property
    def n_decisions(self) -> int:
        return int(round(self.horizon_T / self.decision_dt))

    @property
    def reverse_base(self) -> int:
        """Start index of the REVERSE action block (symmetric mode only)."""
        return 2 * self.max_tasks

    @property
    def action_dim(self) -> int:
        """Flat action layout width (plan Section 3.5), padded to ``max_tasks``.

        Non-symmetric: ``[0, T)`` ACQUIRE, ``[T, 2T)`` DEFEND, then IDLE = ``2T+1``.
        Symmetric (Phase 2): a REVERSE block is inserted — ``[2T, 3T)`` REVERSE task i-2T —
        giving ``3T+1``. Sized to the padding bound so the decoder is config-invariant; the
        non-symmetric width is unchanged so single-team checkpoints load as before.
        """
        return (3 if self.symmetric else 2) * self.max_tasks + 1

    @property
    def idle_action(self) -> int:
        return (3 if self.symmetric else 2) * self.max_tasks

    # ------------------------------------------------------------- construction
    def __post_init__(self) -> None:
        # normalise sequence-typed fields to tuples (YAML gives lists)
        object.__setattr__(self, "field_size", tuple(float(v) for v in self.field_size))
        object.__setattr__(self, "weight_range", tuple(float(v) for v in self.weight_range))
        object.__setattr__(self, "adversary_population", tuple(str(v) for v in self.adversary_population))
        object.__setattr__(self, "holdout_population", tuple(str(v) for v in self.holdout_population))
        self._validate()

    def _validate(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        ratio = self.decision_dt / self.dt
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"decision_dt ({self.decision_dt}) must be a multiple of dt ({self.dt})")
        if self.horizon_T <= 0:
            raise ValueError("horizon_T must be positive")
        for name, count, bound in (
            ("n_robots", self.n_robots, self.max_robots),
            ("n_tasks", self.n_tasks, self.max_tasks),
            ("n_adversaries", self.n_adversaries, self.max_adversaries),
        ):
            if count < 0:
                raise ValueError(f"{name} must be non-negative, got {count}")
            if count > bound:
                raise ValueError(f"{name} ({count}) exceeds its padding bound ({bound})")
        if len(self.field_size) != 2 or any(v <= 0 for v in self.field_size):
            raise ValueError(f"field_size must be two positive values, got {self.field_size}")
        if self.service_radius <= 0:
            raise ValueError("service_radius must be positive")
        for name in ("v_max", "a_max", "adv_v_max", "adv_a_max", "tau_com"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative (0 => tau_rev = inf)")
        if self.reacquire_cost <= 0:
            raise ValueError(f"reacquire_cost (rho) must be positive, got {self.reacquire_cost}")
        if self.stack_cap < 1:
            raise ValueError(f"stack_cap must be >= 1, got {self.stack_cap}")
        if self.stack_value_power <= 0:
            raise ValueError(f"stack_value_power must be positive (1 = linear), got {self.stack_value_power}")
        if self.toggle_multiplier < 1.0:
            raise ValueError(f"toggle_multiplier must be >= 1 (1 = off), got {self.toggle_multiplier}")
        if self.toggle_regions < 0:
            raise ValueError(f"toggle_regions must be >= 0 (0 = off), got {self.toggle_regions}")
        if self.toggle_regions > self.n_tasks:
            raise ValueError(f"toggle_regions ({self.toggle_regions}) cannot exceed n_tasks ({self.n_tasks})")
        if self.dynamic_exposure and self.toggle_regions <= 0:
            raise ValueError("dynamic_exposure requires toggle_regions > 0 (it prices toggles by current cluster load)")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if self.contest_mode not in _CONTEST_MODES:
            raise ValueError(f"contest_mode must be one of {_CONTEST_MODES}, got {self.contest_mode!r}")
        if self.weight_dist not in _WEIGHT_DISTS:
            raise ValueError(f"weight_dist must be one of {_WEIGHT_DISTS}, got {self.weight_dist!r}")
        if self.layout not in _LAYOUTS:
            raise ValueError(f"layout must be one of {_LAYOUTS}, got {self.layout!r}")
        lo, hi = self.weight_range
        if not (0 < lo <= hi):
            raise ValueError(f"weight_range must satisfy 0 < lo <= hi, got {self.weight_range}")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if not (0.0 <= self.protected_fraction <= 1.0):
            raise ValueError(f"protected_fraction must be in [0, 1], got {self.protected_fraction}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalConfig":
        """Build from a plain dict, rejecting any unknown key."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown canonical config keys: {sorted(unknown)}")
        return cls(**data)


def load_config(path: str | Path | None = None, *, overrides: dict[str, Any] | None = None) -> CanonicalConfig:
    """Load and validate a canonical config from YAML.

    The file must contain a top-level ``canonical:`` block. Unknown keys anywhere
    in that block are rejected. ``overrides`` (also strictly checked) are applied
    on top for sweeps.
    """
    path = Path(path) if path is not None else _DEFAULT_PATH
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or "canonical" not in raw:
        raise ValueError(f"{path} must contain a top-level 'canonical:' block")
    block = dict(raw["canonical"])
    if overrides:
        block.update(overrides)
    return CanonicalConfig.from_dict(block)
