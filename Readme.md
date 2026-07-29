# Purifier

Purifier is a Kalman Filter implementation for different systems, written in
Zig on top of [`maryam`](https://github.com/rima1881/maryam). Seven filter
variants:

- `kalman.KalmanFilter` — linear, constant-velocity model.
- `extended_kalman.ExtendedKalmanFilter` — nonlinear model/measurement,
  Jacobian-linearized each step.
- `iterated_extended_kalman.IteratedExtendedKalmanFilter` — same as the EKF,
  but re-linearizes `H` at successively better state estimates within a
  single update instead of just once.
- `unscented_kalman.UnscentedKalmanFilter` — nonlinear model/measurement, no
  derivatives needed (sigma points instead).
- `square_root_kalman.SquareRootKalmanFilter` — same nonlinear model as the
  EKF, but propagates a Cholesky factor of `P` instead of `P` itself.
- `error_state_kalman.ErrorStateKalmanFilter` — same nonlinear model as the
  EKF, but folds each correction onto the nominal state through a
  caller-chosen composition rule instead of always assuming plain vector
  addition.
- `adaptive_kalman.AdaptiveKalmanFilter` — same linear model as `KalmanFilter`,
  but `Q` is re-estimated online from the innovation sequence instead of
  staying at a fixed, hand-picked value for the whole run.

`KalmanFilter`, `ExtendedKalmanFilter`, `IteratedExtendedKalmanFilter`,
`SquareRootKalmanFilter`, and `ErrorStateKalmanFilter` all share the same
covariance math (`kalman_core.KalmanCore`, see `src/kalman_core.zig`);
`UnscentedKalmanFilter` mostly doesn't (sigma-point statistics instead of
`F`/`H`-based propagation — it only reuses `KalmanCore.ApplyGain`, the one
piece that's shape-only and doesn't reference `F`/`H`).
`IteratedExtendedKalmanFilter`, `SquareRootKalmanFilter`, and
`ErrorStateKalmanFilter` all take the exact same `Model` shape as the EKF
(same Jacobians and all — `ErrorStateKalmanFilter` adds two more *optional*
decls on top), and `UnscentedKalmanFilter` takes that same shape minus the
Jacobians, so the same models plug into all five nonlinear filters:

- `examples/ctrv.zig` — CTRV (constant turn rate and velocity) motion model
  plus a radar measurement model, used against the synthetic lidar/radar
  dataset below.
- `examples/gps_ins.zig` — a bicycle model driven by real accelerometer/gyro
  control inputs, used against the real KITTI dataset below.

`zig build run` runs both benchmarks (`imu_bench.zig`, `kitti_bench.zig`),
one after another.

## Filters

- [x] **Linear Kalman Filter** — `kalman.KalmanFilter`
- [x] **Extended Kalman Filter (EKF)** — `extended_kalman.ExtendedKalmanFilter`
- [x] **Unscented Kalman Filter (UKF)** — `unscented_kalman.UnscentedKalmanFilter`.
      Sigma-point propagation instead of Jacobian linearization — `Model`
      only needs `f`/`h`, no derivatives, so every EKF `Model` already
      written (`ctrv.RadarModel`, `gps_ins.GpsModel`) works here unchanged.
      Was blocked on a Cholesky/matrix-square-root primitive; `maryam` added
      `operation.choleskyMatrix` (see `maryam_fix.md` item 8), which unblocked
      it. See the benchmark sections below for results and a real gotcha
      found while wiring it up (initial `P` that's fine for EKF can break
      UKF's sigma points).
- [x] **Square-Root Kalman Filter (SR-KF)** — `square_root_kalman.SquareRootKalmanFilter`.
      Same `Model` interface as the EKF (same Jacobians, every EKF `Model`
      already written works here unchanged), but the persistent state is a
      Cholesky factor `L` of `P` (`P = L @ L^T`) instead of `P` itself.
      `update()` is a genuine Potter/Carlson-form square-root update — it
      propagates `L` directly through `maryam`'s new `operation.qrMatrix`
      (Householder QR) and never re-forms `P` at all, the actual point of
      the technique. `predict()` still round-trips through `P = L @ L^T`,
      for a specific reason: this repo's process noise `Q` (`ctrv.processNoise`,
      `gps_ins.processNoise`) is always rank-deficient by construction (built
      from a low-dimensional noise source), so there's no `Q^0.5` to feed a
      QR-based predict step — not a `maryam` gap, a property of these
      models. See `square_root_kalman.zig`'s doc comment for the full
      recursion and `maryam_fix.md` item 9.
- [x] **Iterated Extended Kalman Filter (IEKF)** — `iterated_extended_kalman.IteratedExtendedKalmanFilter`.
      Re-linearizes `H` at the *updated* state estimate, not just the
      predicted one, iterating within a single update step (the standard
      Gauss-Newton form — Bell & Cathey 1993). Reuses all five of
      `KalmanCore`'s equations, same as `SquareRootKalmanFilter`; only `K`/`S`
      always use the *original* predicted `P`, never an iteration-updated
      one. `max_iterations` is a **comptime** parameter (not a struct field
      like `Q`/`R`), since it's an algorithm-shape choice, not per-instance
      data. A real, somewhat surprising result found while benchmarking
      this: more iterations is **not** monotonically better here — see the
      IMU benchmark section below for the full sweep (1 through 10
      iterations) and why (undamped Gauss-Newton can overshoot and
      oscillate rather than converge smoothly, a known limitation of the
      "vanilla" form without a trust region or line search).
- [x] **Error-State Kalman Filter (ESKF)** — `error_state_kalman.ErrorStateKalmanFilter`.
      Same base `Model` interface as the EKF (`f`/`jacobianF`/`h`/`jacobianH`,
      optional `residual`), and with no further additions it produces
      *exactly* the same `x`/`P` as the EKF at every step (see the
      equivalence test in `error_state_kalman.zig`). What it adds is two more
      optional `Model` decls: `inject(x, dx)`, the "boxplus" composition
      folding a correction `dx` onto the nominal state `x` (default: plain
      `x + dx`), and `resetJacobian(dx)`, the derivative of `inject` at the
      injection point (`G` in the standard derivation — default: identity,
      exact whenever `inject` is linear in `dx`). The textbook motivation is
      orientation state on `SO(3)`/unit quaternions (keeps the error small
      enough for the linearization to stay valid, sidesteps quaternion-
      normalization singularities a direct-state EKF runs into) — this repo
      has no quaternion model to demonstrate that with, so
      `error_state_kalman.zig`'s own tests instead demonstrate the mechanism
      on two simpler cases already present here. First, a periodic state
      component (heading) that `inject` keeps wrapped into `(-pi, pi]` after
      every correction, while an equivalent plain EKF's raw state drifts
      unboundedly — both still converge on the same physical angle, proving
      the wrapping is representational only. Second, a genuinely load-bearing
      one: `ctrv.inject`/`gps_ins.inject` also clamp `v` (forward speed) to
      `>= 0`, a hard physical constraint (neither model's `f` has a
      reverse-gear term) that plain vector addition can't express and the
      EKF has no hook for. On the real, noisy KITTI benchmark below, the
      unconstrained EKF's `v` estimate dips as low as -1.437 — the filter
      pointing velocity 180 degrees off rather than genuinely estimating
      reverse motion — and clamping it measurably helps:
      `ErrorStateKalmanFilter` beats `ExtendedKalmanFilter`'s summed RMSE by
      ~10% on that dataset with the exact same `Q`, no retuning. A third
      test confirms `resetJacobian` is actually applied (not silently
      ignored) by checking its effect on `P` against a hand-computed value.
- [ ] **Information Filter** — the algebraic dual of the standard KF:
      propagates `P^-1` instead of `P`. Natural fit for fusing many
      independent measurement sources, since information from independent
      sensors just adds.
- [x] **Adaptive Kalman Filter** — `adaptive_kalman.AdaptiveKalmanFilter`.
      Every other filter in this repo takes `Q` as a fixed field the caller
      has to hand-pick (see the "untuned" vs. "different Q" columns in the
      IMU benchmark below) — this one re-estimates it online instead, using
      innovation-based adaptive estimation (IAE; Mehra 1970/1972, `Q`
      formula per Mohamed & Schwarz 1999, "Adaptive Kalman Filtering for
      INS/GPS"). Unlike the other five entries on this list, it isn't a new
      model or covariance recursion of its own — it's a wrapper around
      *any one* of the other six filters (see `filter_union.FilterKind`,
      a tagged union over `KalmanFilter`/`ExtendedKalmanFilter`/
      `IteratedExtendedKalmanFilter`/`UnscentedKalmanFilter`/
      `SquareRootKalmanFilter`/`ErrorStateKalmanFilter`), since the
      adaptation only ever needs a step's residual and the gain that mapped
      it onto a state correction, and every variant already computes both
      (Jacobian-based, sigma-point-based, or QR-based) and now exposes them
      as `last_K`/`last_y` fields. `Q` starts at a caller-supplied seed and,
      once a comptime-sized ring buffer of the most recent `window`
      innovations fills, every subsequent `update()` recomputes
      `Q' = K @ Chat @ K^T` from the sample innovation covariance `Chat` and
      the active variant's own `last_K`, before symmetrizing the result the
      same way `unscented_kalman.zig` does (see its own doc comment), and
      writing it back into the active variant's `Q` field — a poorly-guessed
      seed self-corrects instead of staying wrong for the entire run.
      `Kind` and `window` are both **comptime** parameters, not struct
      fields, same reasoning as `IteratedExtendedKalmanFilter`'s
      `max_iterations`: which algorithm is active, and how much history to
      average the estimate over, are algorithm-shape choices, not
      per-instance data. `adaptive_kalman.zig`'s own tests hand-derive the
      exact `Q` a 3-step run converges to through the `.linear` tag, confirm
      the identical mechanism works through the `.ekf` tag on a genuinely
      nonlinear model, and check `Q` stays exactly symmetric through the
      `.ukf` tag's sigma-point-derived gain on a genuinely coupled
      multi-dimensional model — and confirm a badly-underestimated seed `Q`
      (1e-6, against innovations that need it two-plus orders of magnitude
      larger) actually grows to explain the observed data instead of staying
      stuck. **Now wired into both the IMU and KITTI benchmarks below**, unlike
      the unit tests above, on real (or realistically noisy) trajectories —
      and the raw formula above measurably *hurt* accuracy on every dataset
      tried, including a synthetic one built specifically to rule out "not
      enough data" as the explanation. `AdaptiveKalmanFilter` now takes a
      single `Config` struct (only `.window` mandatory, everything else
      defaulted off, so `Config{ .window = w }` alone reproduces the
      original undamped behavior exactly) exposing five further knobs added
      in response, in the order they were tried:
      1) `forgetting_factor` (`b`), blending each estimate with the running
      value (`Q' = (1 - b) * Q + b * estimate`) instead of replacing it
      outright — genuinely helps, but only by approaching "barely adapt at
      all" as `b -> 0`;
      2) `q_floor`/`q_ceiling` (runtime fields), a **diagonal-only**
      approximation of flooring `Q`'s eigenvalues (`maryam` has no general
      eigendecomposition) — found to be actively *dangerous* combined with
      anything but a tiny `b`, since clamping only the diagonal can leave a
      non-positive-semidefinite matrix and cause outright divergence
      (measured: RMSE in the millions);
      3) `diagonal_only`, discarding each estimate's off-diagonal entries
      before blending — fixes exactly that danger, making the floor/ceiling
      an *exact* clamp, at the cost of discarding real correlation
      structure;
      4) `adapt_r`, extending the same windowed statistics to also
      re-estimate `R` (Mohamed & Schwarz's joint Q/R formula) — a clear net
      negative on the lidar/radar dataset (its `R` is well-characterized and
      fixed by construction, so adapting it only adds noise), neutral on the
      GPS-only KITTI dataset;
      5) `burn_in` (skip the first N updates entirely, before the state
      estimate has converged from its wide initial uncertainty) and
      `huber_threshold` (downweight high-Mahalanobis-distance innovations,
      `sqrt(y^T @ S^-1 @ y)`, via every variant's new `last_S` field) —
      **the two genuine, substantial wins**: combined with a still-reactive
      `forgetting_factor = 0.02` (not the near-inert values needed before),
      they cut KITTI's `px` RMSE from 0.8664 down to 0.6215, and on two of
      four components (`py`, `vy`) the fully-tuned configuration now *beats*
      the fixed-`Q` baseline outright. See "Adaptive Kalman Filter: findings"
      under the IMU benchmark section below for the full numbers, sweeps,
      and diagnosis behind each of these five additions.

## Build / run / test

```sh
zig build          # compiles the library + executable
zig build test      # runs the unit tests, including finite-difference checks
                     # on the CTRV Jacobians (see "A real bug" below)
zig build run        # runs all six filter variants against real sensor
                     # data and prints a side-by-side comparison (below)
zig build run -Doptimize=ReleaseFast   # same, but built for speed
```

## Using this as a library

### Adding it as a dependency

```sh
zig fetch --save git+https://github.com/rima1881/Purifier.git
```

```zig
const purifier = b.dependency("Purifier", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("Purifier", purifier.module("Purifier"));
```

### `KalmanFilter(n, k, m)` — linear

`n` = state size, `k` = control size, `m` = measurement size. Set `F`, `B`,
`Q`, `H`, `R` directly as fields; call `predict(u)` then `update(z)` each
step (`update` can fail with `error.SingularMatrix` if the innovation
covariance isn't invertible).

```zig
const std = @import("std");
const Purifier = @import("Purifier");
const kalman = Purifier.kalman;
const extended_kalman = Purifier.extended_kalman;
const iterated_extended_kalman = Purifier.iterated_extended_kalman;
const unscented_kalman = Purifier.unscented_kalman;
const square_root_kalman = Purifier.square_root_kalman;
const error_state_kalman = Purifier.error_state_kalman;
const maryam = @import("maryam");

test "Readme.md: linear KalmanFilter example" {
    // n=2 state [position, velocity], k=1 control (unused here), m=1
    // measurement (position only).
    const KF = kalman.KalmanFilter(2, 1, 1);
    const StateVec = maryam.MatrixType(2, 1);
    const StateMat = maryam.MatrixType(2, 2);
    const ControlVec = maryam.MatrixType(1, 1);
    const ControlMat = maryam.MatrixType(2, 1);
    const MeasureVec = maryam.MatrixType(1, 1);
    const MeasureMat = maryam.MatrixType(1, 2);
    const MeasureNoise = maryam.MatrixType(1, 1);

    var filter = KF{
        .x = StateVec.zero(), // start at position=0, velocity=0
        .P = blk: {
            var m = StateMat.zero();
            m.data[0][0] = 1;
            m.data[1][1] = 1;
            break :blk m;
        },
        .F = blk: { // constant-velocity model, dt=1: position += velocity
            var m = StateMat.zero();
            m.data[0][0] = 1;
            m.data[0][1] = 1;
            m.data[1][1] = 1;
            break :blk m;
        },
        .B = ControlMat.zero(), // no control input in this example
        .Q = StateMat.zero(),
        .H = blk: { // measure position only
            var m = MeasureMat.zero();
            m.data[0][0] = 1;
            break :blk m;
        },
        .R = blk: {
            var m = MeasureNoise.zero();
            m.data[0][0] = 0.1;
            break :blk m;
        },
    };

    filter.predict(ControlVec.zero());

    var z = MeasureVec.zero();
    z.data[0][0] = 1.0; // measured position = 1.0
    try filter.update(z);

    try std.testing.expect(filter.x.data[0][0] > 0); // moved toward the measurement
}
```

### `ExtendedKalmanFilter(n, k, m, Model)` — nonlinear

Same idea, but the model is a **comptime namespace** instead of matrix
fields — `Model` must provide `f`/`jacobianF`/`h`/`jacobianH` (and may
provide `residual`, for measurements that wrap around like an angle; see
`ctrv.radarResidual`).

```zig
test "Readme.md: ExtendedKalmanFilter example" {
    // 1-state EKF estimating x from a nonlinear measurement h(x) = sin(x),
    // linearized (via jacobianH = cos(x)) at the current estimate each step.
    const Vec1 = maryam.MatrixType(1, 1);

    const Model = struct {
        pub fn f(x: Vec1, u: Vec1) Vec1 {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: Vec1, u: Vec1) Vec1 {
            _ = x;
            _ = u;
            var m = Vec1.zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: Vec1) Vec1 {
            var out = Vec1.zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: Vec1) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };

    const EKF = extended_kalman.ExtendedKalmanFilter(1, 1, 1, Model);

    var filter = EKF{
        .x = Vec1.zero(),
        .P = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 1;
            break :blk m;
        },
        .Q = Vec1.zero(),
        .R = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 0.1;
            break :blk m;
        },
    };

    filter.predict(Vec1.zero()); // no control input this step
    var z = Vec1.zero();
    z.data[0][0] = 0.5; // measured sin(x) = 0.5
    try filter.update(z);

    try std.testing.expect(filter.x.data[0][0] > 0); // moved toward asin(0.5)
}
```

### `IteratedExtendedKalmanFilter(n, k, m, Model, max_iterations)` — nonlinear, re-linearized within update()

Same `Model` interface as `ExtendedKalmanFilter` — any EKF `Model` works
here unchanged. `max_iterations` is a 5th **comptime** parameter: starting
from the predicted state, `update()` re-linearizes `H` at successively
better estimates (the standard Gauss-Newton IEKF form) instead of
linearizing once and stopping.

```zig
test "Readme.md: IteratedExtendedKalmanFilter example" {
    // Same model as the EKF example above -- IteratedExtendedKalmanFilter
    // takes the exact same Model interface, so any Model already written
    // for the EKF works here unchanged. `max_iterations` is a 5th
    // **comptime** parameter (not a struct field), since it's an
    // algorithm-shape choice, not per-instance data.
    const Vec1 = maryam.MatrixType(1, 1);

    const Model = struct {
        pub fn f(x: Vec1, u: Vec1) Vec1 {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: Vec1, u: Vec1) Vec1 {
            _ = x;
            _ = u;
            var m = Vec1.zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: Vec1) Vec1 {
            var out = Vec1.zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: Vec1) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };

    const IEKF = iterated_extended_kalman.IteratedExtendedKalmanFilter(1, 1, 1, Model, 3);

    var filter = IEKF{
        .x = Vec1.zero(),
        .P = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 1;
            break :blk m;
        },
        .Q = Vec1.zero(),
        .R = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 0.1;
            break :blk m;
        },
    };

    filter.predict(Vec1.zero()); // no control input this step
    var z = Vec1.zero();
    z.data[0][0] = 0.5; // measured sin(x) = 0.5
    try filter.update(z);

    try std.testing.expect(filter.x.data[0][0] > 0); // moved toward asin(0.5)
}
```

`max_iterations = 1` is exactly the plain EKF update (the predicted state is
also the first, and only, linearization point). Higher values re-linearize
at each new estimate, using a correction term (`- H_i @ (x_pred - x_i)`)
that accounts for linearizing away from the predicted state — see
`iterated_extended_kalman.zig`'s doc comment for the full recursion. **Not
every `Model` benefits equally**: if `jacobianH` doesn't actually depend on
`x` (a linear measurement, like GPS position), every iteration re-linearizes
to the *same* `H`, so iterating changes nothing — see the KITTI benchmark
below, where `IteratedExtendedKalmanFilter` reproduces the plain EKF exactly
for this reason.

### `UnscentedKalmanFilter(n, k, m, Model)` — nonlinear, no derivatives

Same shape as `ExtendedKalmanFilter`, but `Model` only needs `f`/`h` — no
`jacobianF`/`jacobianH`. Instead of linearizing, it propagates a small set
of deterministically-chosen "sigma points" through the actual nonlinear
`f`/`h` and reconstructs mean/covariance from their weighted statistics.
Any `Model` already written for the EKF works here unchanged (the extra
Jacobian decls are just never referenced) — `ctrv.RadarModel`/
`gps_ins.GpsModel` are plugged into all five nonlinear filters in the
benchmarks below.

```zig
test "Readme.md: UnscentedKalmanFilter example" {
    // Same nonlinear measurement as the EKF example above (h(x) = sin(x)),
    // but Model needs no Jacobians -- UKF samples f/h directly at a small
    // set of "sigma points" instead of linearizing around the mean.
    const Vec1 = maryam.MatrixType(1, 1);

    const Model = struct {
        pub fn f(x: Vec1, u: Vec1) Vec1 {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn h(x: Vec1) Vec1 {
            var out = Vec1.zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
    };

    const UKF = unscented_kalman.UnscentedKalmanFilter(1, 1, 1, Model);

    var filter = UKF{
        .x = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 0;
            break :blk m;
        },
        .P = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 1;
            break :blk m;
        },
        .Q = Vec1.zero(),
        .R = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 0.1;
            break :blk m;
        },
    };

    try filter.predict(Vec1.zero()); // no control input this step

    var z = Vec1.zero();
    z.data[0][0] = 0.5; // measured sin(x) = 0.5
    try filter.update(z);

    try std.testing.expect(filter.x.data[0][0] > 0); // moved toward asin(0.5)
}
```

Unlike `ExtendedKalmanFilter`, `predict()` here can fail
(`maryam.EvalError!void`, not `void`): generating sigma points needs a
Cholesky factor of `P`, which only exists if `P` is genuinely symmetric
positive-definite. A **real gotcha** found while wiring this up: an initial
`P` that's perfectly fine for `KalmanFilter`/`ExtendedKalmanFilter` (which
only ever use `P` algebraically, never sample points from it) can badly
break the UKF, if it's "wide" on a periodic state component like an angle —
see the KITTI benchmark section below for the actual bug this caused and the
fix.

### `SquareRootKalmanFilter(n, k, m, Model)` — nonlinear, Cholesky-factored `P`

Same `Model` interface as `ExtendedKalmanFilter` (same `f`/`jacobianF`/`h`/
`jacobianH`, same optional `residual`) — any EKF `Model` works here
unchanged. The only difference is the field: `L`, a Cholesky factor of the
covariance (`P = L @ L^T`), instead of `P` itself.

```zig
test "Readme.md: SquareRootKalmanFilter example" {
    // Same model and Jacobians as the ExtendedKalmanFilter example above --
    // SquareRootKalmanFilter takes the exact same Model interface, so any
    // Model already written for the EKF works here unchanged. The only
    // difference is the field: `L` (a Cholesky factor of P, P = L @ L^T)
    // instead of `P` itself.
    const Vec1 = maryam.MatrixType(1, 1);

    const Model = struct {
        pub fn f(x: Vec1, u: Vec1) Vec1 {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: Vec1, u: Vec1) Vec1 {
            _ = x;
            _ = u;
            var m = Vec1.zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: Vec1) Vec1 {
            var out = Vec1.zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: Vec1) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };

    const SRKF = square_root_kalman.SquareRootKalmanFilter(1, 1, 1, Model);

    var filter = SRKF{
        .x = Vec1.zero(),
        .L = blk: { // P = L @ L^T = 1, same starting uncertainty as the EKF example
            var m = Vec1.zero();
            m.data[0][0] = 1;
            break :blk m;
        },
        .Q = Vec1.zero(),
        .R = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 0.1;
            break :blk m;
        },
    };

    try filter.predict(Vec1.zero()); // no control input this step
    var z = Vec1.zero();
    z.data[0][0] = 0.5; // measured sin(x) = 0.5
    try filter.update(z);

    try std.testing.expect(filter.x.data[0][0] > 0); // moved toward asin(0.5)
}
```

Like the UKF, `predict()`/`update()` here return `maryam.EvalError!void`, not
`void`: `predict()` re-factors `P` back into `L` via
`operation.choleskyMatrix`, and `update()` calls `operation.qrMatrix` (both
can fail — `error.NotPositiveDefinite` and `error.RankDeficient`
respectively). `update()` is a genuine Potter/Carlson-form square-root
update: it builds an augmented `(m+n) x (m+n)` block matrix from `R`'s own
Cholesky factor, `H @ L`, and `L`, QR-decomposes it, and reads the new gain
and the new `L` straight out of blocks of the resulting upper-triangular
factor — `P` itself is never reconstructed, the actual point of the
technique (keeps the working condition number at `cond(L)` instead of
`cond(P) = cond(L)^2`). `predict()` still round-trips through
`P = L @ L^T`, because this repo's process noise `Q` is always
rank-deficient by construction (see the checklist entry above) — there's no
`Q^0.5` to feed an equivalent QR-based predict step for these models. In
exact arithmetic both forms compute the identical answer regardless, and
the benchmark sections below confirm it: `SquareRootKalmanFilter` still
matches `ExtendedKalmanFilter`'s RMSE to 4 decimal places on both datasets.
See `square_root_kalman.zig`'s doc comment for the full recursion and why
`L`'s sign after `update()` isn't guaranteed to match `predict()`'s
convention (harmless — `L` is only ever used via `L @ L^T` or `H @ L`).

**The benchmark datasets never actually get ill-conditioned enough to show
the difference, so `square_root_kalman.zig` has its own tests that
deliberately do.** Both use a classic square-root-filter stress scenario: a
2-state system with no dynamics, observed only through a fixed linear
combination (`H = [1, 1]`, measuring `a + b`) with tiny measurement noise —
genuine partial observability (like fusing redundant sensors), not an
artificial pathological matrix. The "sum" direction's variance keeps
shrinking every update while the "difference" direction (never observed)
stays put, driving `cond(P)` up every single step.

- At `R = 1e-12` after 100 updates (verified independently against a
  50-decimal-digit `mpmath` reference implementing the identical recursion):
  the plain Joseph-form `KalmanFilter`'s small eigenvalue is off by **~45%**
  from the true value — it's numerically "stuck" and no longer tracking the
  covariance once `cond(P)` gets this large. `SquareRootKalmanFilter`'s is
  off by **~2.3%** — not perfect, but a genuine ~20x reduction in error,
  from the exact same recursion, computed via QR instead of squaring `P`
  directly. This is the actual, concrete version of the folklore claim
  square-root filters are built on.
- At `R = 1e-31`, `SquareRootKalmanFilter` reports `error.RankDeficient` on
  the very first `update()` call — `R`'s own Cholesky factor is at the edge
  of `f64`'s relative precision compared to the rest of the augmented QR
  matrix, and `maryam`'s `qrMatrix` has an explicit tolerance check for
  exactly this. The plain Joseph-form filter has no equivalent check and
  runs to completion regardless, not because it's more capable at this
  level of ill-conditioning, but because nothing in its formula ever asks
  whether the result is still numerically meaningful.

### `ErrorStateKalmanFilter(n, k, m, Model)` — nonlinear, caller-chosen state composition

Same `Model` interface as `ExtendedKalmanFilter` (same `f`/`jacobianF`/`h`/
`jacobianH`, same optional `residual`) — any EKF `Model` works here
unchanged, and with no further additions this filter produces *exactly* the
same `x`/`P` as `ExtendedKalmanFilter` at every step.

```zig
test "Readme.md: ErrorStateKalmanFilter example" {
    // Same model and Jacobians as the ExtendedKalmanFilter example above --
    // ErrorStateKalmanFilter takes the exact same Model interface, so any
    // Model already written for the EKF works here unchanged, and behaves
    // identically to it unless Model also supplies inject/resetJacobian.
    const Vec1 = maryam.MatrixType(1, 1);

    const Model = struct {
        pub fn f(x: Vec1, u: Vec1) Vec1 {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: Vec1, u: Vec1) Vec1 {
            _ = x;
            _ = u;
            var m = Vec1.zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: Vec1) Vec1 {
            var out = Vec1.zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: Vec1) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };

    const ESKF = error_state_kalman.ErrorStateKalmanFilter(1, 1, 1, Model);

    var filter = ESKF{
        .x = Vec1.zero(),
        .P = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 1;
            break :blk m;
        },
        .Q = Vec1.zero(),
        .R = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 0.1;
            break :blk m;
        },
    };

    filter.predict(Vec1.zero()); // no control input this step
    var z = Vec1.zero();
    z.data[0][0] = 0.5; // measured sin(x) = 0.5
    try filter.update(z);

    try std.testing.expect(filter.x.data[0][0] > 0); // moved toward asin(0.5)
}
```

What it adds over the plain EKF is two more *optional* `Model` decls, used
inside `update()` to fold the Kalman correction `dx = K @ y` onto the
nominal state instead of always assuming flat vector addition:

- `inject(x: StateVec, dx: StateVec) StateVec` — the "boxplus" composition
  `x_new = x (+) dx`. Defaults to plain `x + dx` (identical to the EKF).
- `resetJacobian(dx: StateVec) StateMat` — the Jacobian of `inject` at the
  injection point (`G` in the standard ESKF derivation — e.g. Sola 2017,
  "Quaternion kinematics for the error-state Kalman filter", §6), applied
  as `P' = G @ Joseph(...) @ G^T` after the ordinary Joseph-form update.
  Defaults to the identity, which is exact whenever `inject` is linear
  in `dx`.

The textbook motivation is orientation state on `SO(3)`/unit quaternions:
`inject` composes a small rotation onto the nominal quaternion and
renormalizes, keeping the error small enough for the linearization to stay
valid and sidestepping the singularities a direct-state EKF hits trying to
treat a quaternion as a flat vector. This repo has no quaternion model to
demonstrate that with, so `error_state_kalman.zig`'s own tests demonstrate
the same mechanism on the simplest case already present here instead: a
1-state heading tracker where `inject` wraps the result into `(-pi, pi]`
after every correction. After 500 predict+update cycles, the
`ErrorStateKalmanFilter`'s stored state never leaves that range, while an
otherwise-identical plain `ExtendedKalmanFilter`'s raw additive state drifts
past `2*pi` — and wrapping the EKF's state lands on the ESKF's, within
estimation error, confirming the wrapping changes only the *representation*,
not what's being estimated. A third test checks a deliberately non-identity
`resetJacobian` actually scales `P` by the expected amount, rather than
being silently ignored.

### `AdaptiveKalmanFilter(n, k, m, Kind, config)` — wraps any other filter, online `Q`/`R` estimation

Unlike every other filter above, this one isn't a new model or covariance
recursion — it's a wrapper around *any one* of the other six, via
`filter_union.FilterKind(n, k, m, Model, iekf_iterations)`, a tagged union
over `KalmanFilter`/`ExtendedKalmanFilter`/`IteratedExtendedKalmanFilter`/
`UnscentedKalmanFilter`/`SquareRootKalmanFilter`/`ErrorStateKalmanFilter`,
all monomorphized for the same `(n, k, m, Model)`. `Kind` and `config`
(a `adaptive_kalman.Config`, a **comptime** value — which algorithm is
active and how the estimator behaves are algorithm-shape choices, matching
`IteratedExtendedKalmanFilter`'s own `max_iterations`) are both comptime
parameters. `Config` has one mandatory field and five optional ones, all
defaulted off:

```zig
pub const Config = struct {
    window: usize,                    // innovations averaged into each estimate
    forgetting_factor: f64 = 1.0,     // Q'/R' = (1-b)*old + b*estimate; 1.0 = replace outright
    burn_in: usize = 0,               // update() calls to skip before the ring buffer starts filling
    diagonal_only: bool = false,      // discard off-diagonal terms before blending
    adapt_r: bool = false,            // also re-estimate R, not just Q
    huber_threshold: ?f64 = null,     // downweight innovations past this Mahalanobis distance
};
```

`Config{ .window = w }` alone reproduces the original, undamped IAE formula
exactly (every hand-derived test written before the other fields existed
still passes unchanged). See "Adaptive Kalman Filter: findings" below for
what each of the other five actually does to real accuracy, including one
that's actively dangerous in combination (`q_floor`/`q_ceiling` +
non-`diagonal_only` + a non-tiny `forgetting_factor`) and one that's a clear
net negative for at least one of the two datasets tested (`adapt_r`).

```zig
test "Readme.md: AdaptiveKalmanFilter example" {
    // Same h(x) = sin(x) model as the ExtendedKalmanFilter example above,
    // wrapped in AdaptiveKalmanFilter via filter_union.FilterKind's .ekf tag
    // instead of driven standalone -- proves the online-Q mechanism isn't
    // special-cased to the linear filter. Q starts badly wrong (1e-6, far
    // too small); window=5 means Q gets re-estimated from the actual
    // innovation sequence once 5 updates have run.
    const Vec1 = maryam.MatrixType(1, 1);

    const Model = struct {
        pub fn f(x: Vec1, u: Vec1) Vec1 {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: Vec1, u: Vec1) Vec1 {
            _ = x;
            _ = u;
            var m = Vec1.zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: Vec1) Vec1 {
            var out = Vec1.zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: Vec1) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };

    const Kind = filter_union.FilterKind(1, 1, 1, Model, 1);
    const AKF = adaptive_kalman.AdaptiveKalmanFilter(1, 1, 1, Kind, .{ .window = 5 });

    var filter = AKF{
        .active = .{ .ekf = extended_kalman.ExtendedKalmanFilter(1, 1, 1, Model){
            .x = Vec1.zero(),
            .P = blk: {
                var m = Vec1.zero();
                m.data[0][0] = 1;
                break :blk m;
            },
            .Q = blk: { // deliberately too small -- the point of this filter
                var m = Vec1.zero();
                m.data[0][0] = 1e-6;
                break :blk m;
            },
            .R = blk: {
                var m = Vec1.zero();
                m.data[0][0] = 0.1;
                break :blk m;
            },
        } },
    };

    const seed_q = filter.active.ekf.Q.data[0][0];

    var i: usize = 0;
    while (i < 20) : (i += 1) {
        try filter.predict(Vec1.zero());
        var z = Vec1.zero();
        z.data[0][0] = if (i % 2 == 0) 0.9 else -0.9; // sustained, large innovations
        try filter.update(z);
    }

    // After enough updates, Q has been re-estimated from the actual
    // innovation sequence instead of staying at the badly-guessed seed.
    try std.testing.expect(filter.active.ekf.Q.data[0][0] > seed_q * 100);
}
```

`active` holds whichever concrete filter struct is currently driving
`predict()`/`update()`, constructed exactly the way it would be used
standalone (same fields, same seed `Q`) — this wrapper adds no state of its
own beyond the innovation ring buffer. `predict()`/`update()` both return
`maryam.EvalError!void` regardless of which tag is active, even for the four
variants whose own `predict()` can't fail (`.linear`/`.ekf`/`.iekf`/`.eskf`)
— a uniform signature across all six, since `active`'s tag is only known at
runtime. Note that because of that same runtime dispatch, `Model` has to be
a genuine, full EKF-compatible namespace (`f`/`jacobianF`/`h`/`jacobianH`)
even if the caller only ever intends to use the `.linear` tag: every switch
arm has to compile, not just the one actually reached. See
`filter_union.zig`'s doc comment for why, and `adaptive_kalman.zig`'s own
tests for the identical mechanism exercised through `.linear`, `.ekf`, and
`.ukf` (the last confirming a sigma-point-derived gain plugs into the same
formula, and that `Q` stays exactly symmetric on a genuinely coupled
multi-dimensional model, without any UKF-specific handling in this filter).

Six more *runtime* fields round out the picture: `q_floor`/`q_ceiling`
(`Core.StateVec`, one bound per state component, default `0`/`+inf` — a
no-op unless set) clamp `Q`'s diagonal after every blend, and
`r_floor`/`r_ceiling` (`Core.MeasureVec`, same idea) do the same for `R`
when `Config.adapt_r` is set. All four exist for the same reason
`forgetting_factor` does: the raw formula measurably hurt accuracy on real
trajectories, and these (plus `Config.burn_in`/`diagonal_only`/
`huber_threshold`) are the fixes that followed. See "Adaptive Kalman Filter:
findings" below for the full investigation, including the real trap:
`q_floor`/`q_ceiling` only clamp the diagonal (`maryam` has no general
eigendecomposition to floor the true eigenspectrum with), and combining
that with a non-tiny `forgetting_factor` while *not* also setting
`Config.diagonal_only` can produce an invalid (non-positive-semidefinite)
`Q` and cause outright divergence — not just a weaker fix, a genuinely worse
one. The benchmarks below use
`Config{ .window = 20, .forgetting_factor = 0.02, .burn_in = 20,
.huber_threshold = 1.5 }`, the configuration that investigation actually
converged on.

Every code block above (EKF, IEKF, UKF, SR-KF, ESKF, Adaptive KF) is copied
verbatim from `examples/readme_examples.zig`.

For a full worked model (multi-dimensional state, real Jacobians, process
noise from control-input uncertainty, angle-wrapped residuals) see
`examples/ctrv.zig` or `examples/gps_ins.zig` rather than reimplementing one
from scratch — both are plugged into all five nonlinear filters in the
benchmarks below. `ctrv.LidarModel`/`ctrv.RadarModel` and `gps_ins.GpsModel`
now also supply `inject`/`resetJacobian`, used only by
`ErrorStateKalmanFilter` — every other filter type these `Model`s already
plug into looks up `f`/`jacobianF`/`h`/`jacobianH`/`residual` by name and
never sees the two extra decls, so adding them changed nothing about the
EKF/IEKF/UKF/SR-KF rows below (confirmed: their RMSE numbers are unchanged
from before ESKF existed). Their `inject` does two things: wraps `yaw` into
`(-pi, pi]` (a no-op on both benchmarks below — see the IMU/KITTI sections
for why), and clamps `v` (forward speed) to `>= 0`, since neither model's
`f` has a reverse-gear term. That second one is *not* a no-op on the real,
noisy KITTI data: the unconstrained EKF's `v` estimate genuinely swings
negative there, and clamping it measurably improves accuracy — see the
KITTI accuracy section below for the actual numbers.

## The IMU benchmark (`zig build run`)

`examples/imu_bench.zig` replays all six filters against the same real
(synthetic-but-sensor-realistic) lidar/radar dataset and scores each against
ground truth.

**Data source**: [`obj_pose-laser-radar-synthetic-input.txt`](https://github.com/udacity/CarND-Extended-Kalman-Filter-Project/blob/master/data/obj_pose-laser-radar-synthetic-input.txt),
from Udacity's Extended Kalman Filter project — a bicycle-model (CTRV)
vehicle trajectory with simulated lidar (`px, py`) and radar (`rho, theta,
rho_dot`) measurements plus ground-truth `(px, py, vx, vy)` at every step.
Vendored at `examples/data/laser_radar_synthetic.txt` (250 lidar rows, 250 radar
rows).

The linear `KalmanFilter(4, 1, 2)` (state `[px, py, vx, vy]`) can only
consume the lidar rows — radar's `rho_dot` is a nonlinear function of the
state, which a linear filter can't express. `ExtendedKalmanFilter(5, 1, m)`,
`IteratedExtendedKalmanFilter(5, 1, m, Model, 3)`,
`UnscentedKalmanFilter(5, 1, m)`, `SquareRootKalmanFilter(5, 1, m)`, and
`ErrorStateKalmanFilter(5, 1, m)` all use a 5-state CTRV model
(`[px, py, v, yaw, yaw_rate]`, `examples/ctrv.zig`) with a genuinely
nonlinear radar measurement model, so all five consume *all* 500 rows —
same `ctrv.LidarModel`/`ctrv.RadarModel` plugged into each.

### Accuracy

```
-- Linear KF -- constant-velocity model, lidar only --
RMSE      px=0.1211  py=0.0986  vx=0.4818  vy=0.4576
max |err| px=0.3415  py=0.2876  vx=2.6175  vy=1.2662

-- Extended KF (untuned Q) -- CTRV model, lidar + radar --
RMSE      px=0.0700  py=0.0803  vx=0.2142  vy=0.3011
max |err| px=0.2073  py=0.2896  vx=2.1940  vy=3.4988

-- Extended KF (different Q) -- CTRV model, lidar + radar --
RMSE      px=0.0591  py=0.0828  vx=0.1900  vy=0.2878
max |err| px=0.1806  py=0.2904  vx=2.1743  vy=3.5031

-- Iterated EKF, max 3 iterations (same Q as untuned EKF) -- CTRV model, lidar + radar --
RMSE      px=0.0701  py=0.0826  vx=0.1819  vy=0.2128
max |err| px=0.2077  py=0.2899  vx=0.8147  vy=1.2705

-- Unscented KF (same Q as untuned EKF) -- CTRV model, lidar + radar --
RMSE      px=0.0717  py=0.0838  vx=0.2406  vy=0.2414
max |err| px=0.3073  py=0.2877  vx=1.7392  vy=1.7527

-- Square-Root KF (same Q as untuned EKF) -- CTRV model, lidar + radar --
RMSE      px=0.0700  py=0.0803  vx=0.2142  vy=0.3011
max |err| px=0.2073  py=0.2896  vx=2.1940  vy=3.4988

-- Error-State KF (same Q as untuned EKF) -- CTRV model, lidar + radar --
RMSE      px=0.0700  py=0.0803  vx=0.2142  vy=0.3011
max |err| px=0.2073  py=0.2896  vx=2.1940  vy=3.4988

-- Adaptive EKF, window=20 (Q starts at untuned seed, then self-estimated) -- CTRV model, lidar + radar --
RMSE      px=0.0725  py=0.0851  vx=0.2188  vy=0.3433
max |err| px=0.2678  py=0.2867  vx=2.1940  vy=3.4988
```

| RMSE | Linear KF (lidar only) | EKF, untuned Q | EKF, different Q | IEKF, untuned Q\*\*\*\* | UKF, untuned Q | SR-KF, untuned Q\*\*\* | ESKF, untuned Q\*\*\*\*\* | Adaptive EKF\*\*\*\*\*\* | Reference C++ EKF (CV, lidar+radar)\* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| px | 0.1211 | 0.0700 | **0.0591** | 0.0701 | 0.0717 | 0.0700 | 0.0700 | 0.0725 | 0.0972 |
| py | 0.0986 | 0.0803 | 0.0828 | 0.0826 | 0.0838 | 0.0803 | 0.0803 | 0.0851 | **0.0854** |
| vx | 0.4818 | 0.2142 | 0.1900 | **0.1819** | 0.2406\*\* | 0.2142 | 0.2142 | 0.2188 | 0.4509 |
| vy | 0.4576 | 0.3011 | 0.2878 | **0.2128** | 0.2414 | 0.3011 | 0.3011 | 0.3433 | 0.4396 |

\*\*\*\*\*\* `Config{ .window = 20, .forgetting_factor = 0.02, .burn_in = 20,
.huber_threshold = 1.5 }` — still worse than "EKF, untuned Q" here (px/py by
~4-6%, vx/vy by ~2-14%, starting from that *exact same* seed), but far
closer than the raw formula's original ~4-10x. See "Adaptive Kalman Filter:
findings" below for the full progression across five added fixes, and for
the KITTI dataset below where two of four components now *beat* the fixed
baseline outright.

\*\*\*\*\* Identical to "EKF, untuned Q" to 4 decimal places, unlike the
SR-KF column this isn't a numerical-equivalence result — `ErrorStateKalmanFilter`
runs the exact same arithmetic `ExtendedKalmanFilter` does, plus two extra
steps in `ctrv.LidarModel`/`ctrv.RadarModel`'s `inject`: wrapping `yaw` into
`(-pi, pi]`, and clamping `v` to `>= 0`. Both are no-ops here — this
dataset's real `yaw` trajectory never gets near a multiple of `2*pi`, and
(unlike the KITTI benchmark below) `v` never actually goes negative on this
synthetic dataset. See `error_state_kalman.zig`'s own tests (and the API
section above) for `yaw`-wrapping actually mattering, and the KITTI accuracy
section below for `v`-clamping actually mattering.

\*\*\*\* **IEKF beats every other config (including the hand-tuned EKF) on
`vx`/`vy`, using the untuned `Q`** — and its `max|err|` on `vx`/`vy`
(`0.8147`/`1.2705`) is dramatically tighter than the plain EKF's
(`2.1940`/`3.4988`), meaning it doesn't just average better, it avoids the
EKF's worst-case radar-linearization spikes. But this specific
`max_iterations = 3` was chosen *empirically* — see the discussion below,
this is not a "more iterations = better" story.

\*\*\* Identical to "EKF, untuned Q" to 4 decimal places, exactly as
expected: `SquareRootKalmanFilter`'s `update()` is now a genuine QR-based
square-root update (see the API section above), but it's still
algebraically the same recursion as the EKF's, just computed via a
different, condition-number-preserving route -- see the validation note
below.

\*\* UKF beats every EKF configuration on `vy`, and beats the *untuned* EKF
on `vx` (though not the hand-tuned one, or IEKF) — using the exact same,
untuned `Q` as the "EKF, untuned Q" column. No Jacobians were derived for
this model at all; the sigma points handle CTRV's `sin`/`cos` nonlinearity
directly.

**A real, somewhat surprising result: iterating more is not monotonically
better.** `imu_bench.zig`'s `LidarIEKF`/`RadarIEKF` use `max_iterations = 3`
specifically because a full sweep (1 through 10) turned out non-monotonic:

| max_iterations | 1 (= plain EKF) | 2 | 3 | 4 | 5 | 8 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RMSE vx | 0.2142 | 0.3730 | **0.1819** | 0.5227 | 0.5599 | 0.5827 | 0.3856 |
| RMSE vy | 0.3011 | 0.1825 | **0.2128** | 0.3429 | 0.6003 | 0.3894 | 0.3199 |

3 iterations is the best performer of the sweep; every value beyond it (4,
5, 8) is *worse than doing no iterating at all* (`max_iterations = 1`). This
is a known property of "vanilla" Gauss-Newton IEKF, not a bug: with no step
damping or line search, re-linearizing at an overshot estimate can produce a
*worse* next estimate than the one before it, so the sequence oscillates
around a solution instead of settling into it monotonically. (Verified this
wasn't a Zig-side quirk — the exact same non-monotonic RMSE values at every
iteration count reproduce in the independent Python validation described
below.) The practical takeaway: `max_iterations` for a vanilla IEKF is a
tuning knob to sweep empirically per problem, not a dial where "higher is
safer."

The IEKF row is independently validated the same way the SR-KF row is: no
off-the-shelf library (filterpy included) has an IEKF class, so the standard
Gauss-Newton recursion (Bell & Cathey 1993) is hand-transcribed in Python,
independently of `iterated_extended_kalman.zig`'s own code, and applied with
the same CTRV `f`/`h`/Jacobians. Result at `max_iterations = 3`: `RMSE
px=0.0701 py=0.0826 vx=0.1819 vy=0.2128`, `max|err|` also identical — a
**bit-for-bit match**. The full 1/2/4/5 sweep matches too, confirming the
non-monotonic behavior above is a genuine property of the algorithm on this
data, not an implementation quirk.

The UKF row is independently validated against
[`filterpy`](https://github.com/rlabbe/filterpy)'s `UnscentedKalmanFilter`
(`MerweScaledSigmaPoints(n=5, alpha=1.0, beta=2.0, kappa=0.0)` — this repo's
exact fixed sigma-point parameters — same untuned `Q`/`R`, same CTRV `f`/`h`
and radar angle-wrap residual, two filter objects kept in sync on shared
`(x, P)` to mirror `imu_bench.zig` alternating lidar/radar rows): RMSE
`px=0.0718 py=0.0839 vx=0.2407 vy=0.2423`, max|err| identical to the table
above on all four components. That's close but not the bit-for-bit match the
KITTI bicycle UKF gets against the same library (see below) — over 499
sequential nonlinear CTRV steps (vs. KITTI's 153, on a less nonlinear
model), two independently-written Cholesky decompositions (this repo's
direct Crout-style factorization vs. `scipy.linalg.cholesky`'s LAPACK
routine) accumulate floating-point rounding differently, and CTRV's
straight-line/turning branch (`|yaw_rate| > 1e-4`) means a rounding-sized
state difference can occasionally flip which branch a given step takes.
Two independent implementations still agreeing to 3-4 significant figures on
every RMSE component, with an *exact* match on every max|err|, is the useful
result here: it rules out a structural bug (wrong sign, wrong weight, wrong
sigma-point formula), which is what this check is actually for — bit-for-bit
equivalence isn't a bar two independently-written nonlinear recursive
filters should be expected to clear.

The SR-KF row is validated differently, since filterpy's own
`SquareRootKalmanFilter` class is linear-only (plain `F`/`H` attributes, no
nonlinear-model hook the way its `ExtendedKalmanFilter` has `predict_x`/
`HJacobian`/`Hx`) — it can't be pointed at a nonlinear model directly. So the
check instead hand-transcribes the exact QR-based (Potter-form) recursion
straight from filterpy's own `square_root.py` source and applies it with
this repo's CTRV `f`/`h` and Jacobians: `RMSE px=0.0700 py=0.0803 vx=0.2142
vy=0.3011`, `max|err|` also identical — a **bit-for-bit match**, unlike the
UKF's. That's expected: this repo's own `SquareRootKalmanFilter.update()` is
now the same kind of QR-based square-root update filterpy's is (both via
Householder QR — `maryam`'s `operation.qrMatrix` on the Zig side), and this
model has no branch as sensitive to rounding as the UKF's sigma points, so
both implementations land on exactly the same floating-point trajectory.

`imu_bench.zig` runs both the untuned defaults (`ekf_std_a_untuned = 2.0`,
`ekf_std_yawdd_untuned = 0.5` — a reasonable guess, not fit to the data) and
a different `Q` (`ekf_std_a_alt = 0.5`, `ekf_std_yawdd_alt = 0.5`, found
with a coarse grid search over this exact dataset, minimizing summed RMSE
across all four components) side by side, rather than replacing one with
the other — so the effect of changing `Q` is visible instead of hidden.
The different `Q` improves `px`/`vx`/`vy` (`px` RMSE drops another 15% past
the reference implementation below) but very slightly *worsens* `py`
(0.0803 → 0.0828) — a reminder that minimizing the summed RMSE doesn't mean
improving every component, especially with correlated state (position and
velocity share process noise here).

[atul799/CarND-Extended-Kalman-Filter-Project](https://github.com/atul799/CarND-Extended-Kalman-Filter-Project,
`main_no_simulator.cpp`, built with Eigen 3.4.0 via `zig c++`), using the
canonical 4-state constant-velocity model (same as this repo's linear
filter) plus an EKF radar update bolted on — same `R_laser`/`R_radar`/
`noise_ax`/`noise_ay` values as this repo uses, so the noise assumptions
line up. It also independently normalizes the radar bearing residual into
`[-pi, pi]` — the exact same fix `ctrv.radarResidual` implements here,
arrived at separately, which is a good sign it's a real, load-bearing
requirement and not something specific to this codebase.

Two things stand out:

1. **Our EKF beats it on every metric**, `vx`/`vy` by roughly 2x. The
   reference's CV model has the same weakness this repo's own linear filter
   has: it doesn't turn. Adding radar fixed its lack of a second sensor but
   not its motion model, so the same turn-induced velocity lag that hurts
   the linear filter here still hurts it there. Our CTRV model doesn't have
   that problem, and *also* consumes radar, so it gets both advantages while
   the reference gets only one.
2. **Our linear filter (lidar only, no radar at all) still beats the
   reference's `vx`/`vy`**, and is close on `px`/`py` despite having half
   the measurements. Radar's `rho_dot` helps, but a correctly-modeled motion
   turns out to matter more than a second sensor here.

The EKF, IEKF, UKF, SR-KF, and ESKF all win over the linear filter, as they
should: they see twice the measurements (radar's `rho_dot` speaks to
velocity directly, which lidar never does) and CTRV actually turns, instead
of assuming constant-velocity straight-line motion like the linear filter
does.

All six EKF/IEKF/UKF/SR-KF/ESKF columns above use the exact same
`ctrv.RadarModel`/`ctrv.LidarModel` — only `Q` differs between the two EKF
columns, IEKF re-linearizes `H` up to 3 times per update instead of once,
the filter engine (Jacobians vs. sigma points) differs for the UKF column,
the covariance representation (`P` directly vs. its Cholesky factor `L`)
differs for the SR-KF column, and the ESKF column additionally runs every
correction through `ctrv.inject` (a no-op on this data — see the footnote
above) instead of plain addition (see `imu_bench.zig`).

### Adaptive Kalman Filter: findings

`AdaptiveKalmanFilter`'s own unit tests (see the API section above) prove
the online-`Q` mechanism works — a badly-wrong seed genuinely self-corrects.
Run against real (or realistically noisy) trajectories instead of a
hand-picked toy scenario, the raw formula doesn't help. On *every* dataset
tried below, across a wide sweep of `window` sizes, it did worse than just
leaving `Q` at the untuned seed and never touching it again — often
dramatically worse.

The obvious first suspicion, given how the previous investigations in this
README went (IEKF's iteration count, UKF's initial `P`), was "these datasets
are too small for a sliding-window estimator to get a reliable read." That
turned out to be the wrong explanation. Two separate datasets were swept:

| window | 5 | 10 | 15 | 20 | 30 | 50 | 75 | 200 (never adapts) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KITTI (153 steps) RMSE px | 1.0170 | 1.0232 | 1.0385 | 1.0391 | 1.0091 | 0.8966 | 0.7936 | 0.5117 |

(KITTI's "EKF, untuned Q" baseline is `px=0.5156` — see the accuracy table
below. `window=200` exceeds the 153-step run, so `Q` never leaves its seed;
its `px=0.5117` essentially reproduces the untuned baseline, confirming the
harness itself is correct and the degradation at every *smaller* window is
really coming from the adaptation, not a wiring bug.)

| window | 10 | 20 | 30 | 50 | 100 | 249 (barely adapts) |
| --- | --- | --- | --- | --- | --- | --- |
| CTRV, 500 rows, RMSE vx | 7.3673 | 3.5560 | 3.1321 | 3.2645 | 2.0458 | 0.2145 |
| CTRV, 500 rows, RMSE vy | 8.5785 | 3.2593 | 2.2605 | 2.2296 | 1.7663 | 0.3013 |

(this dataset's "EKF, untuned Q" baseline is `vx=0.2142 vy=0.3011` — up to
**~28x worse** at `window=10`. `window=249` is almost the entire 250-row
lidar/radar stream each sensor sees, so `Q` barely ever gets recomputed;
that it converges back toward the baseline as `window` grows is the tell
that this is a real property of the adaptation, not noise.)

To rule out "the datasets are just too small" directly rather than arguing
it from the trend above, a third, purpose-built dataset was generated:
`examples/scripts/make_long_ctrv_dataset.py` synthesizes 5000 rows (10x the
existing CTRV dataset) of a CTRV vehicle cycling through eight
straight/turning legs at varying speed, with lidar/radar noise matching the
existing benchmark's own `R` assumptions — saved as
`examples/data/ctrv_long_synthetic.txt`. Ten times the data did not fix it:

| window | 10 | 20 | 30 | 50 | 100 | 249 |
| --- | --- | --- | --- | --- | --- | --- |
| RMSE px | 7.2406 | 7.6835 | 5.7599 | 6.7243 | 4.7088 | 0.8816 |

against a baseline of `px=0.0782` (this dataset's own "EKF, untuned Q") —
still roughly **11x worse** even at `window=249`, out of 2500 lidar rows
available. More data made the *absolute* numbers worse, not better; the
only thing that improved them was making `window` large enough that
adaptation triggers less often — the opposite of what an online estimator
is supposed to buy you.

**Diagnosis**: the original `Q' = K @ Chat @ K^T` is the textbook Mohamed &
Schwarz (1999) formula, implemented with no damping whatsoever — no
forgetting factor blending the new estimate with the old one, no floor
preventing `Q` from collapsing toward (or growing away from) a sane range,
and a small sliding window is, by construction, a noisy sample estimate of
a covariance. On a real (or realistically noisy) nonlinear trajectory with
actual turning/accelerating legs, a chunk of the innovation sequence's
variance comes from genuine transient tracking error and CTRV linearization
mismatch at each turn, not from `Q` being wrong — and the raw estimator
can't tell the difference, so it misattributes that variance to `Q` and
inflates or shrinks it accordingly, feeding a worse `Q` into the *next*
`predict()`. This is a known, published failure mode of "vanilla" IAE
without a forgetting factor (see e.g. Mohamed & Schwarz's own follow-up
discussion, and the broader adaptive-KF literature's general preference for
damped/blended variants over the raw sliding-window estimator).

**Fix attempted: a forgetting-factor blend and a diagonal floor/ceiling.**
`AdaptiveKalmanFilter` gained two comptime/runtime knobs (see the API
section above): `forgetting_factor` (`b`), blending
`Q' = (1 - b) * Q + b * estimate` instead of replacing `Q` outright
(`b = 1` reproduces the original formula exactly), and `q_floor`/
`q_ceiling`, clamping `Q`'s diagonal after every blend. Sweeping `b` (at
`window=20`, no floor/ceiling) on all three datasets:

| b | 1 (original) | 0.3 | 0.1 | 0.05 | 0.02 | 0.01 | 0.005 | 0.002 | 0.001 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRV, 500 rows, RMSE px | 0.2959 | 0.2373 | 0.2715 | 0.0997 | 0.0756 | 0.0744 | 0.0726 | 0.0708 | 0.0700 |
| CTRV, 500 rows, RMSE vy | 3.2593 | 7.8893 | 3.4762 | 1.0812 | 0.3605 | 0.3225 | 0.3107 | 0.3045 | 0.3026 |
| KITTI, 153 steps, RMSE px | 1.0391 | — | — | — | 0.8664 | 0.7792 | 0.7039 | 0.6267 | 0.5846 |
| CTRV-long, 5000 rows, RMSE px | 7.6835\* | — | 1.2964 | — | 0.1567 | — | 0.0895 | — | 0.0836 |

(\*`b=1` row for the long dataset is `window=20`'s original-formula result,
reported earlier in this section as `7.6835` at that window.) This
genuinely works, in the sense that it's a real, large, monotonic
improvement — CTRV's `vy` alone drops from **~10.8x** the untuned baseline
(`0.3011`) at `b=1` to **~1.005x** at `b=0.001`, and the long dataset's `px`
drops from ~98x its own baseline (`0.0782`) at `b=1` to ~1.07x at
`b=0.001`. But look at *where* the improvement comes from: it's monotonic
in `b`, and the best results sit at the smallest `b` values tried — the
blend isn't finding a sweet spot where genuine adaptation outperforms a
fixed `Q`, it's approaching the limit `b -> 0`, i.e. *stop adapting*.
`b=0.001` on 500 rows means each window-full `update()` (roughly one every
20 steps) nudges `Q` by 0.1% — after ~25 such nudges over the whole run,
`Q` has barely moved from its seed. No `b` tested on any dataset beats the
untuned fixed baseline on every component; the closer a configuration gets
to matching it, the less the filter is actually doing. KITTI's shorter
153-step run recovers less than the two 500+ row datasets even at the same
tiny `b` (`px=0.5846` vs. baseline `0.5156`, still ~13% worse) — fewer total
window-refreshes means less opportunity for the early, noisiest estimates to
get diluted out.

**The floor/ceiling turned out to be actively dangerous, not just
ineffective**, when combined with anything but a very small `b`. Testing
`q_ceiling=0.1` (roughly 10x this dataset's own seed `Q`'s largest diagonal
entry) alongside moderate `b`:

| b | 0.3 | 0.1 | 0.05 | 0.02 |
| --- | --- | --- | --- | --- |
| CTRV, RMSE vx (no ceiling) | — | — | 0.5752 | 0.2281 |
| CTRV, RMSE vx (q_ceiling=0.1) | 7,633,019 | 390 | 0.5752 | 0.2281 |
| KITTI, RMSE vx (q_ceiling=0.1) | 133.76 | 1603.88 | 1.5394 | 1.2574 |

At `b=0.3`/`0.1`, clamping the diagonal produces RMSE in the *thousands to
millions* — genuine catastrophic divergence, not a rounding-sized
regression. The reason is exactly what `adaptive_kalman.zig`'s own doc
comment warns about: `q_floor`/`q_ceiling` only clamp `Q`'s diagonal, not
its full eigenspectrum, because `maryam` has no general eigendecomposition
to floor the real thing with. When `b` is large enough that a single noisy
window can swing `Q`'s *off-diagonal* terms to something large while the
diagonal gets clamped down, the result can violate positive-semidefiniteness
outright (`|Q[i][j]| > sqrt(Q[i][i] * Q[j][j])`) — an invalid covariance
that later linear algebra (matrix inversion inside `S^-1`) has no obligation
to handle gracefully. The clamp is only safe when `b` is already small
enough that `Q` doesn't swing wildly in the first place — at which point it
mostly has nothing left to do.

**Round two: `diagonal_only`, `adapt_r`, `burn_in`, `huber_threshold`.**
Four more fixes were tried, gated behind their own `Config` fields (see the
API section above). Two didn't help; two genuinely did.

`Config.diagonal_only` fixes the floor/ceiling danger directly, by
discarding each estimate's off-diagonal terms before blending — a diagonal
matrix's eigenvalues *are* its diagonal, so the clamp becomes exact instead
of an approximation that can produce an invalid matrix. Confirmed: the same
`q_ceiling=0.1` configuration that produced millions-scale RMSE above no
longer diverges at all with `diagonal_only` set, at any `b` tried:

| b | 0.1 | 0.3 | 1.0 (original formula, but now diagonal + clamped) |
| --- | --- | --- | --- |
| CTRV, RMSE vx (diagonal_only + q_ceiling=0.1) | 0.3008 | 0.3362 | 0.3623 |

No more divergence — but also not better than plain small-`b` damping
without `diagonal_only` (`vx=0.2281` at `b=0.02`, no clamp at all). The real
cost is exactly what it sounds like: discarding legitimate correlation
structure (e.g. `ctrv.processNoise`'s cross terms between position and
velocity) along with the noise.

`Config.adapt_r` extends the same windowed statistics to `R`, via Mohamed &
Schwarz's joint formula (`R' = Chat - (Sbar - R)`, using every variant's new
`last_S` field — see `kalman.zig`'s doc comment). On the lidar/radar
dataset it's a clear net negative — `px=3.06 vx=33.1` at `window=20,
b=0.02, adapt_r=true`, wildly worse than without it (`px=0.0756 vx=0.2281`)
— because this dataset's `R` is a fixed, accurately-known sensor
specification by construction; adapting it just adds a second noisy
estimator with nothing real to correct. On KITTI (single GPS sensor, `R`
arguably *more* plausible to be wrong) it's closer to neutral
(`px=0.8707` vs. `0.8664` without it) — not a win, but not the same kind of
disaster.

`Config.burn_in` and `Config.huber_threshold` are the two genuine wins.
`burn_in` skips the first N `update()` calls entirely rather than letting
them into the ring buffer — the state estimate hasn't converged from its
wide initial `P` yet, so the very first would-be window is the least
representative data available, and with no forgetting factor small enough
to fully protect against it, that first bad estimate does damage later
windows have to dilute back out. `huber_threshold` downweights (not
hard-excludes) individual buffered innovations by their Mahalanobis
distance `sqrt(y^T @ S^-1 @ y)` — a more targeted version of the same idea,
aimed specifically at the turning/accelerating-transient spikes the
diagnosis above blames for the raw formula's instability. Swept
individually and combined, at the same reactive `window=20, b=0.02` used
throughout (not shrunk toward the "barely adapt" limit):

| config | CTRV RMSE px | CTRV RMSE vx | KITTI RMSE px | KITTI RMSE vx |
| --- | --- | --- | --- | --- |
| baseline (`b=0.02` alone) | 0.0756 | 0.2281 | 0.8664 | 1.1830 |
| `burn_in=20` | 0.0735 | 0.2157 | 0.6459 | 0.9995 |
| `huber_threshold=1.5` | 0.0735 | 0.2287 | 0.8127 | 1.1347 |
| `burn_in=20` + `huber_threshold=1.5` | 0.0725 | 0.2188 | 0.6215 | 0.9962 |

Both help individually, and combined help more than either alone — a real,
non-trivial improvement at a `b` that's still doing genuine, reactive
adaptation, not just approaching a no-op. KITTI benefits far more than
CTRV: `burn_in` alone cuts its `px` by 25% (`0.8664 -> 0.6459`), consistent
with the diagnosis that a short 153-step run is disproportionately damaged
by one bad early window with little later data to dilute it back out.
Pushing `forgetting_factor` down toward the earlier sweep's near-inert
values *on top of* `burn_in`/`huber_threshold` closes the remaining gap
almost entirely — at `window=20, b=0.001, burn_in=20, huber_threshold=1.5`:
CTRV `px=0.0698` (very slightly *beating* the untuned baseline's `0.0700`)
and KITTI `px=0.5196` (within 0.8% of the baseline's `0.5156`, with `py`/
`vy` on that dataset actually beating it — see the accuracy table below).
Stacking `adapt_r` on top of that combination changes essentially nothing
(`px=0.5197`, identical to four decimal places) — at a `b` this small,
there's almost no adaptation left for it to destabilize.

**The upshot**: `AdaptiveKalmanFilter` is correct throughout — every one of
its six behaviors (the original IAE estimator, the forgetting-factor blend,
the diagonal clamp, `diagonal_only`, `adapt_r`, `burn_in`, and
`huber_threshold`) is backed by a hand-derived unit test proving it computes
exactly the formula it claims to. Two additions (`burn_in`,
`huber_threshold`) are genuine, substantial improvements even at a
meaningfully reactive forgetting factor, not just a way to approach "stop
adapting." One (`adapt_r`) is dataset-dependent, ranging from a clear net
negative to roughly neutral. One (`diagonal_only`) trades real correlation
structure for making the floor/ceiling safe rather than dangerous. And the
fully-combined configuration reported in the benchmarks below
(`Config{ .window = 20, .forgetting_factor = 0.02, .burn_in = 20,
.huber_threshold = 1.5 }`) is close enough to a well-chosen fixed `Q` — on
two of eight tracked components (KITTI's `py`/`vy`) it now genuinely beats
it — that the honest summary has shifted from the original "this actively
hurts, don't use it" to "with real engineering effort past the textbook
formula, this is roughly competitive with a fixed `Q` on real data, without
ever needing that fixed `Q` hand-picked in the first place." Whether that
trade is worth the added complexity and the one dataset-dependent knob
(`adapt_r`) that can still hurt is a genuine judgment call, not a settled
one.

### Speed

Timed around `predict()` + `update()` only (excludes text parsing), native
target:

| filter | build | ns/cycle | cycles/sec |
| --- | --- | --- | --- |
| Linear KF | Debug | ~5020-5410 | ~185-199k |
| Linear KF | ReleaseFast | ~122-161 | ~6.2-8.2M |
| Extended KF (either Q) | Debug | ~7100-7640 | ~131-141k |
| Extended KF (either Q) | ReleaseFast | ~245-346 | ~2.9-4.1M |
| Iterated EKF (3 iterations) | Debug | ~11640-11750 | ~85-86k |
| Iterated EKF (3 iterations) | ReleaseFast | ~418-429 | ~2.3-2.4M |
| Unscented KF | Debug | ~27490-27930 | ~35.8-36.4k |
| Unscented KF | ReleaseFast | ~510-623 | ~1.6-2.0M |
| Square-Root KF | Debug | ~8100-8130 | ~123-124k |
| Square-Root KF | ReleaseFast | ~657-832 | ~1.2-1.5M |
| Error-State KF | Debug | ~8340-8440 | ~118-120k |
| Error-State KF | ReleaseFast | ~273-342 | ~2.9-3.7M |
| Adaptive EKF (window=20) | Debug | ~18690-19450 | ~51-54k |
| Adaptive EKF (window=20) | ReleaseFast | ~945-953 | ~1.05-1.06M |

The EKF costs ~1.4x (Debug) to ~2.5x (ReleaseFast) the linear filter per
cycle — Jacobian evaluation plus a 5-state (vs. 4-state) covariance
recursion, plus radar's update runs against a 3-row measurement instead of
lidar's 2. `Q`'s *value* doesn't change any of this — same matrix size,
same equations, just different numbers going in. IEKF (3 iterations) costs
~1.6-2.2x the plain EKF, roughly proportional to the iteration count, since
each pass re-evaluates `jacobianH`/`h` and re-solves for `K` — cheap
individually (no Cholesky, no sigma points), but it's real work done 3
times instead of once. UKF costs another ~2.2-2.5x over the EKF: `2n+1=11`
sigma points each get propagated through `f`/`h` individually every
`predict()`/`update()`, plus a Cholesky decomposition per `predict()` —
genuinely more arithmetic, not implementation overhead. SR-KF costs
roughly the same as the EKF in Debug, but noticeably more in ReleaseFast
(~3x, not the ~20-30% it cost before switching `update()` to a real
QR-based recursion): `update()`'s Householder QR on an `(m+n) x (m+n)`
matrix is a comparable amount of raw arithmetic to the old Joseph-form
path, but it's a tighter loop over a dynamically-sized reflector vector
that the optimizer evidently handles less well than the old path's plain
matrix-multiply calls. The tradeoff is deliberate: `update()` gets the
technique's actual numerical benefit now (never re-forming `P`), at a real
speed cost in the fast build specifically. ESKF costs about the same as the
plain EKF in both builds — it runs the identical Joseph-form recursion, plus
one extra vector-add-sized step (`inject`) that's within noise here.
Adaptive EKF costs roughly ~2.5x the plain EKF in ReleaseFast and
noticeably more in Debug (~2.4x) — every `update()` runs the plain EKF
recursion plus per-step ring-buffer bookkeeping (now including a
Mahalanobis-distance evaluation per buffered entry for
`huber_threshold`'s weighting, itself a small matrix solve), and once every
`window=20` steps also rebuilds `Chat` from all 20 buffered innovations,
recomputes `Q'`, blends it with the running `Q` (`forgetting_factor`), and
clamps the result; that periodic cost amortizes to a modest per-step
average, but the per-step Mahalanobis evaluation (needed for
`huber_threshold`) runs unconditionally, not just on window-refresh steps,
which is most of this column's cost increase over earlier in this
README's history. Real, extra work — that, per the accuracy findings above,
gets close to (and on some components beats) a well-chosen fixed `Q`, but
only after adding several fixes past the original textbook formula.

### Memory

```
Linear KF struct       = 656 bytes (n=4, k=1, m=2 state)
Extended KF struct     = 728 bytes (n=5, k=1, m=2 or 3 state)
Iterated EKF struct    = 728 bytes (n=5, k=1, m=2 or 3 state) -- identical fields to the EKF's; max_iterations is comptime, not stored
Unscented KF struct    = 1168 bytes (n=5, k=1, m=2 or 3 state; carries 11 cached sigma points between predict() and update())
Square-Root KF struct  = 728 bytes (n=5, k=1, m=2 or 3 state) -- identical to the EKF's: same fields (x, Q, R), just P renamed to L
Error-State KF struct  = 728 bytes (n=5, k=1, m=2 or 3 state) -- identical fields to the EKF's; inject/resetJacobian are Model decls, not stored state
Adaptive EKF struct    = 3248 bytes (wraps a FilterKind union over all six of the above, sized to the largest -- here the UKF's 1168 bytes -- plus a tag, q_floor/q_ceiling/r_floor/r_ceiling, and two 20-entry ring buffers: innovations and their innovation covariances)
process peak RSS       = ~4.4-6.1 MB (whole program: runtime + embedded dataset + all filters)
```

Every filter's struct is somewhat larger than earlier in this README's
history: `last_K`/`last_y`/`last_S` (see `kalman.zig`'s doc comment) were
added to every non-adaptive filter specifically so `AdaptiveKalmanFilter`
could read back "the gain, residual, and innovation covariance this step
actually used" generically, regardless of which algorithm produced them.
`AdaptiveKalmanFilter` itself is the largest by a wide margin because a
`union(enum)` is sized to fit its biggest variant (the UKF's, since it also
carries cached sigma points) plus a tag, on top of its own two
`window=20`-entry ring buffers (innovations and their covariances, `m=2` or
`3` each) and four floor/ceiling bound vectors.

No filter ever touches an allocator — every `maryam` matrix is a plain
`[rows][cols]f64` stack value, so each filter's persistent footprint is just
its own `@sizeOf(...)`, a fixed compile-time constant regardless of how long
it runs. Process peak RSS is dominated by the Zig runtime and the ~12KB
embedded dataset, not by any one filter.

## The KITTI benchmark (also `zig build run`)

`examples/kitti_bench.zig` runs a different comparison against a real (not
synthetic) dataset: [KITTI raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php),
sequence `2011_09_26_drive_0005`, `oxts` channel only (GPS/IMU, from an OXTS
RT3003 GPS/INS unit) — 154 frames, ~15.4s of a car turning through roughly
75 degrees. `oxts`'s own output is treated as ground truth position/speed/
heading. `examples/data/kitti_gps_imu.txt` adds synthetic Gaussian position noise
(fixed seed, std=2m, typical consumer-GPS accuracy) on top of the true
trajectory to simulate a GPS receiver, and carries the real forward-
acceleration (`af`) and yaw-rate (`wu`) channels unmodified.

This is a structurally different problem from the lidar/radar one above:
one sensor (GPS position), not two, so the interesting question isn't "can
this filter consume a nonlinear sensor" — it's "does using the real IMU as
a control input (dead-reckoning between GPS fixes) beat assuming constant
velocity." `kalman.KalmanFilter` never touches the IMU signals: pure
constant-velocity, corrected by GPS. `gps_ins.GpsModel`'s bicycle model
propagates state using the actual accelerometer/gyro readings each step
(`v' = v + af·dt`, `yaw' = yaw + wu·dt`), then corrects with the same GPS.

### Accuracy

```
-- Linear KF -- constant-velocity model, GPS only (no IMU) --
RMSE      px=0.6342  py=0.9290  vx=0.8059  vy=1.5129
max |err| px=1.9998  py=2.0974  vx=2.1275  vy=12.4841

-- Bicycle EKF (untuned Q) -- IMU-driven, GPS correction --
RMSE      px=0.5156  py=0.9568  vx=0.9840  vy=1.3867
max |err| px=1.5164  py=2.1841  vx=6.3497  vy=6.8624

-- Bicycle EKF (different Q) -- IMU-driven, GPS correction --
RMSE      px=0.4233  py=0.9343  vx=1.0038  vy=1.3689
max |err| px=1.5205  py=2.2020  vx=6.3495  vy=6.8623

-- Bicycle IEKF, max 3 iterations (same Q as untuned EKF) -- IMU-driven, GPS correction --
RMSE      px=0.5156  py=0.9568  vx=0.9840  vy=1.3867
max |err| px=1.5164  py=2.1841  vx=6.3497  vy=6.8624

-- Bicycle UKF (same Q as untuned EKF) -- IMU-driven, GPS correction --
RMSE      px=0.6856  py=1.1894  vx=0.7344  vy=1.1420
max |err| px=1.8594  py=2.9322  vx=2.0484  vy=3.2594

-- Bicycle SR-KF (same Q as untuned EKF) -- IMU-driven, GPS correction --
RMSE      px=0.5156  py=0.9568  vx=0.9840  vy=1.3867
max |err| px=1.5164  py=2.1841  vx=6.3497  vy=6.8624

-- Bicycle ESKF (same Q as untuned EKF) -- IMU-driven, GPS correction --
RMSE      px=0.5767  py=0.7372  vx=0.9113  vy=1.2419
max |err| px=1.7074  py=2.1841  vx=6.3497  vy=6.8624

-- Bicycle Adaptive EKF, window=20 (Q starts at untuned seed, then self-estimated) -- IMU-driven, GPS correction --
RMSE      px=0.6215  py=0.9544  vx=0.9962  vy=1.3829
max |err| px=1.7892  py=2.1841  vx=6.3497  vy=6.8624
```

| RMSE | Linear KF (GPS only) | EKF, untuned Q | EKF, different Q | IEKF, untuned Q\*\*\*\* | UKF, untuned Q\*\* | SR-KF, untuned Q\*\*\* | ESKF, untuned Q\*\*\*\*\* | Adaptive EKF\*\*\*\*\*\* | filterpy reference (EKF)\* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| px | 0.6342 | 0.5156 | **0.4233** | 0.5156 | 0.6856 | 0.5156 | 0.5767 | 0.6215 | 0.5156 |
| py | 0.9290 | 0.9568 | 0.9343 | 0.9568 | 1.1894 | 0.9568 | **0.7372** | 0.9544 | 0.9568 |
| vx | 0.8059 | 0.9840 | 1.0038 | 0.9840 | **0.7344** | 0.9840 | 0.9113 | 0.9962 | 0.9840 |
| vy | 1.5129 | 1.3867 | 1.3689 | 1.3867 | **1.1420** | 1.3867 | 1.2419 | 1.3829 | 1.3867 |

\*\*\*\*\*\* `Config{ .window = 20, .forgetting_factor = 0.02, .burn_in = 20,
.huber_threshold = 1.5 }` — same configuration as the IMU benchmark above.
`py` and `vy` here actually *beat* "EKF, untuned Q" (`0.9544` vs. `0.9568`,
`1.3829` vs. `1.3867`) despite starting from that exact same seed `Q`; `px`
and `vx` are close but still slightly worse (`0.6215` vs. `0.5156`, `0.9962`
vs. `0.9840`). A real improvement over the raw formula's original result on
this dataset (`px=1.0391`), driven mostly by `burn_in`: this dataset's short
153-step run means the very first ring-buffer window, built from the
least-converged early state estimate, previously did disproportionate
damage that never got diluted out by later, better data — see "Adaptive
Kalman Filter: findings" in the IMU benchmark section above, which covers
both benchmarks in full.

\*\*\*\*\* **Unlike every other footnoted column in either benchmark, this
one is *not* identical to "EKF, untuned Q"** — same `Q`, same model, same
data, and the numbers genuinely differ: summed RMSE drops from `0.5156 +
0.9568 + 0.9840 + 1.3867 = 3.8431` to `0.5767 + 0.7372 + 0.9113 + 1.2419 =
3.4671`, a **~10% reduction**, driven almost entirely by `py` (`0.9568 →
0.7372`). The reason: `gps_ins.GpsModel`'s `inject` clamps `v` (forward
speed) to `>= 0` (see the API section above and `gps_ins.inject`'s doc
comment) — a hard constraint the plain EKF has no way to enforce, since it
only ever adds the Kalman correction and stops. On this real, noisy 154-frame
sequence the *unconstrained* EKF's `v` estimate actually goes as low as
`-1.437` (measured directly, not estimated), which isn't "the car is
inching backward" — `gps_ins.f`'s `px' = px + v*cos(yaw)*dt` has no
reverse-gear term, so a negative `v` means the filter is pointing the
velocity vector 180 degrees off and lying about the magnitude to
compensate, which then poisons every subsequent `predict()`. `yaw`-wrapping
is still a no-op here, same reasoning as the IMU benchmark's ESKF footnote
(this trajectory's ~75 degrees of real turning never approaches a multiple
of `2*pi`) — the entire improvement comes from the `v` clamp. `px`'s max
`|err|` is the one number that gets *worse* (`1.5164 → 1.7074`): clamping
`v` to exactly `0` instead of leaving it at whatever value produced the
least-squares-optimal (if unphysical) correction is a real tradeoff, not a
free lunch — it's a net win here, but "net" is doing real work in that
sentence.

\*\*\*\* **Identical to "EKF, untuned Q" to 4 decimal places, but for a
different reason than the SR-KF column**: `gps_ins.GpsModel`'s measurement
is GPS position, `h(x) = [px, py]`, whose Jacobian `jacobianH` is a constant
matrix (`[[1,0,0,0],[0,1,0,0]]`) that doesn't depend on `x` at all. IEKF's
entire mechanism is re-linearizing `H` at successively better state
estimates — with a constant `H`, every iteration re-linearizes to the exact
same matrix, so the second and later iterations are no-ops (the early-exit
tolerance fires after one pass) and the result collapses to exactly the
plain EKF. This is the honest flip side of the IMU benchmark's IEKF result
below, where radar's `H` genuinely depends on `x` and iterating measurably
helps: IEKF's value is entirely tied to how state-dependent the
measurement's Jacobian actually is.

\*\*\* Identical to "EKF, untuned Q" to 4 decimal places, same reasoning as
the IMU benchmark's SR-KF column above: `update()` is a genuine QR-based
square-root update now, but still algebraically the same recursion as the
EKF's.

\* [`filterpy`](https://github.com/rlabbe/filterpy) (Roger Labbe's widely-used
open-source Kalman filter library, `pip install filterpy`) implementing the
identical bicycle model with the **untuned** noise values — same equations,
same process/measurement noise — as an independent check. It matches this
repo's Zig implementation to 4 decimal places on every metric, which is the
useful result of this comparison: the two independent implementations agree
exactly, so the untuned numbers are a property of the model/data, not an
implementation bug. (The "different Q" column has no `filterpy` counterpart
— it's only validating correctness, not chasing the same result.)

\*\* Same story on this dataset as the lidar/radar one above: with the exact
same (untuned) `Q` as the EKF, sigma-point propagation through the bicycle
model wins `vx`/`vy` outright and loses `px`/`py` to the tuned EKF — no
Jacobian was derived for this. `filterpy`'s `UnscentedKalmanFilter`
(`MerweScaledSigmaPoints(n=4, alpha=1.0, beta=2.0, kappa=0.0)`, matching this
repo's fixed sigma-point parameters exactly) reproduces every one of these
numbers to 4 decimal places, which is what actually caught the bug below.

The SR-KF column is validated the same way the IMU benchmark's is: since
filterpy's own `SquareRootKalmanFilter` can't be pointed at a nonlinear
model, its exact QR-based (Potter-form) recursion is hand-transcribed from
`filterpy/kalman/square_root.py` and applied with `gps_ins.zig`'s bicycle
`f`/Jacobians instead. Result: `RMSE px=0.5156 py=0.9568 vx=0.9840
vy=1.3867`, `max|err|` also identical — a bit-for-bit match with both this
repo's own SR-KF and the plain EKF, exactly as expected (see the IMU
benchmark's SR-KF validation note for why this one matches exactly while the
UKF's doesn't).

The IEKF column needs no separate numerical validation beyond the EKF's own
`filterpy` match above: since `gps_ins.GpsModel`'s `H` is constant (see the
table footnote), IEKF is provably identical to the EKF on this model for any
correct implementation, and the Zig output matching the EKF's `filterpy`
number to 4 decimal places already confirms that. The IMU benchmark's IEKF
validation note (hand-transcribed Gauss-Newton recursion, independently
matching to 4 decimal places on the CTRV model where iteration actually
does something) is the one that actually exercises IEKF's own logic.

The ESKF column is the one exception to "identical to the EKF" among the
footnoted columns here, and it's validated differently as a result: not by
matching the EKF, but by the `inject clamps v to 0...` tests in
`gps_ins.zig`, which pin down exactly when the clamp fires (deterministically,
on hand-picked `x`/`dx` values) independently of this benchmark's specific
noisy trajectory. What *is* still true, and worth stating plainly, is that
`ErrorStateKalmanFilter` runs `ExtendedKalmanFilter`'s identical Joseph-form
recursion for everything except the one extra `inject` call — the ~10%
improvement isn't a different, more sophisticated estimator, it's the exact
same estimator with one physically-motivated constraint bolted onto the
correction step. See the accuracy table footnote above for the actual
numbers and reasoning.

### A real bug: UKF sigma points need a real initial covariance

The bicycle EKF above initializes velocity/yaw-rate variance wide-open
(`P[2][2] = P[3][3] = 1000`) since it has no prior belief about them and `P`
that large is harmless for the EKF/linear filters — they only ever use `P`
algebraically (as an uncertainty ellipsoid in matrix equations), never
sample a point that far from the mean.

Copying that same `P` into the UKF's init produced catastrophic divergence
(`px` RMSE ~1.16, `vx` RMSE ~6.19 — worse than every other filter by a wide
margin). The reason: UKF's sigma points are genuinely sampled at
`mean ± sqrt(n+lambda)*L_column`, and `sqrt(1000)` times this filter's
spread factor (~2) is ~63. For `yaw` — a periodic quantity — 63 radians
wraps around 2π about ten times, landing sigma points at essentially
arbitrary headings instead of "a plausible nearby heading." The fix was
using `P[2][2] = P[3][3] = 1` for the UKF's own init only, leaving the EKF's
init untouched. The `filterpy` exact-match above was what confirmed the fix
was correct, not just "no longer obviously broken."

`kitti_bench.zig` runs two `Q` configurations side by side rather than
picking one: the untuned defaults above (`ekf_std_af_untuned = 1.0`,
`ekf_std_wu_untuned = 0.2`, kept exactly as originally chosen so the
`filterpy` validation stays reproducible), and a different `Q`
(`ekf_std_af_alt = 0.15`, `ekf_std_wu_alt = 0.03`, found with a coarse grid
search over this exact dataset, minimizing summed RMSE).

Two things are worth noting:

1. **Changing `Q` moves the numbers far less here than it did for the CTRV
   filter above** (summed RMSE drops ~3%, vs. ~8% there). With only ~154
   real, noisy samples and jittery real sensor timestamps, there's less
   signal for a different `Q` to exploit than in the longer synthetic
   lidar/radar dataset.
2. **There still isn't a clean winner** — the EKF wins `px`/`vy`, the
   linear filter still wins `py`/`vx` (and the different `Q` made `vx`
   slightly *worse*, 0.9840 → 1.0038). That's a real result, not a bug (the
   untuned case reproduces exactly in `filterpy`), and is worth taking at
   face value: a smarter motion model doesn't automatically dominate a
   dumber one on a short, real, noisy sequence the way it did on the longer
   synthetic one. That's a more honest picture of real-world EKF gains than
   the lidar/radar benchmark's clean sweep.

### Speed & memory

| filter | build | ns/cycle | struct bytes |
| --- | --- | --- | --- |
| Linear KF | Debug | ~4810-4910 | 656 |
| Linear KF | ReleaseFast | ~104-116 | 656 |
| Bicycle EKF (either Q) | Debug | ~4470-4790 | 432 |
| Bicycle EKF (either Q) | ReleaseFast | ~122-143 | 432 |
| Bicycle IEKF (3 iterations) | Debug | ~6490-6860 | 432 |
| Bicycle IEKF (3 iterations) | ReleaseFast | ~160-167 | 432 |
| Bicycle UKF | Debug | ~18320-18540 | 720 |
| Bicycle UKF | ReleaseFast | ~227-237 | 720 |
| Bicycle SR-KF | Debug | ~5070-5160 | 432 |
| Bicycle SR-KF | ReleaseFast | ~376-381 | 432 |
| Bicycle ESKF | Debug | ~5090-5400 | 432 |
| Bicycle ESKF | ReleaseFast | ~126-128 | 432 |
| Bicycle Adaptive EKF (window=20) | Debug | ~12330-12590 | 1808 |
| Bicycle Adaptive EKF (window=20) | ReleaseFast | ~448-482 | 1808 |

The bicycle EKF's struct is smaller than the linear filter's (432 vs. 656
bytes) despite being an EKF, because its state is the same size (`n=4`) but
it doesn't carry persistent `F`/`B`/`H` fields the way the linear filter
does — `F` is recomputed from the Jacobian each step and never stored (every
filter's struct here is larger than earlier in this README's history:
`last_K`/`last_y`/`last_S`, added so `AdaptiveKalmanFilter` can read back
each variant's gain, residual, and innovation covariance generically, cost
every one of them an extra `n x m` matrix, `m`-vector, and `m x m` matrix).
The UKF's struct (720 bytes) is bigger than either: it caches all
`2n+1 = 9` sigma points between `predict()` and `update()` so `update()`
doesn't need to regenerate them from `P` a second time. The SR-KF's struct
is the same 432 bytes as the EKF's (same fields, `P` just renamed to `L`).
Its Debug speed sits close to the EKF's, but ReleaseFast is ~3x the EKF's
now that `update()` runs a real Householder QR instead of a Joseph-form
recursion (same tradeoff as the IMU benchmark above — real numerical
benefit, real ReleaseFast cost). IEKF's struct is also the same 432 bytes
(`max_iterations` is comptime, not a stored field), but unlike the IMU
benchmark, its speed here is barely above the plain EKF's (~1.1-1.4x, not
the ~1.6-2.2x there) — the 3 iterations still run, but with a constant `H`
(see the accuracy table footnote above) each pass after the first is doing
real but wasted linear-algebra work on an answer that's already converged.
ESKF's struct is the same 432 bytes too (`inject`/`resetJacobian` are
`Model` decls, not stored fields) and its speed matches the plain EKF's
closely in both builds, same reasoning as the IMU benchmark above: it's the
identical recursion plus one `inject` call that's a no-op here. The Adaptive
EKF's struct (1808 bytes) is the biggest by far: a `union(enum)` over all
six other filters sized to its largest member (the UKF's 720 bytes) plus a
tag, four floor/ceiling bound vectors (`q_floor`/`q_ceiling`/`r_floor`/
`r_ceiling`), and two `window=20`-entry ring buffers (innovations and their
covariances, needed for `huber_threshold`'s Mahalanobis weighting and
`adapt_r`'s joint estimation). Its ReleaseFast speed (~3.5-3.9x the plain
EKF, up from the ~1.1x an earlier, simpler version of this filter had) is
now dominated by the per-step Mahalanobis-distance evaluation
`huber_threshold` requires unconditionally, not just the periodic
`Chat`/`Q'`/`R'` recomputation every `window` steps.
