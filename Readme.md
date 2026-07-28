# Purifier

Purifier is a Kalman Filter implementation for different systems, written in
Zig on top of [`maryam`](https://github.com/rima1881/maryam). Six filter
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
- [ ] **Adaptive Kalman Filter** (online process-noise estimation) — every
      filter in this repo currently uses hand-picked, untuned process noise
      (see the "untuned" notes in both benchmarks below); this would
      estimate it from the innovation sequence instead of guessing.

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

Every code block above (EKF, IEKF, UKF, SR-KF, ESKF) is copied verbatim from
`examples/readme_examples.zig`.

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
```

| RMSE | Linear KF (lidar only) | EKF, untuned Q | EKF, different Q | IEKF, untuned Q\*\*\*\* | UKF, untuned Q | SR-KF, untuned Q\*\*\* | ESKF, untuned Q\*\*\*\*\* | Reference C++ EKF (CV, lidar+radar)\* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| px | 0.1211 | 0.0700 | **0.0591** | 0.0701 | 0.0717 | 0.0700 | 0.0700 | 0.0972 |
| py | 0.0986 | 0.0803 | 0.0828 | 0.0826 | 0.0838 | 0.0803 | 0.0803 | **0.0854** |
| vx | 0.4818 | 0.2142 | 0.1900 | **0.1819** | 0.2406\*\* | 0.2142 | 0.2142 | 0.4509 |
| vy | 0.4576 | 0.3011 | 0.2878 | **0.2128** | 0.2414 | 0.3011 | 0.3011 | 0.4396 |

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

### Speed

Timed around `predict()` + `update()` only (excludes text parsing), native
target:

| filter | build | ns/cycle | cycles/sec |
| --- | --- | --- | --- |
| Linear KF | Debug | ~5200-5500 | ~181-193k |
| Linear KF | ReleaseFast | ~82-84 | ~12M |
| Extended KF (either Q) | Debug | ~7600-8000 | ~125-131k |
| Extended KF (either Q) | ReleaseFast | ~206-208 | ~4.8M |
| Iterated EKF (3 iterations) | Debug | ~12600-17600 | ~57-79k |
| Iterated EKF (3 iterations) | ReleaseFast | ~380-383 | ~2.6M |
| Unscented KF | Debug | ~17500-20000 | ~50-57k |
| Unscented KF | ReleaseFast | ~579-600 | ~1.7M |
| Square-Root KF | Debug | ~8200-8750 | ~114-122k |
| Square-Root KF | ReleaseFast | ~635-765 | ~1.3-1.6M |
| Error-State KF | Debug | ~8700-9050 | ~110-115k |
| Error-State KF | ReleaseFast | ~294-304 | ~3.3-3.4M |

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

### Memory

```
Linear KF struct       = 544 bytes (n=4, k=1, m=2 state)
Extended KF struct     = 512 bytes (n=5, k=1, m=2 or 3 state)
Iterated EKF struct    = 512 bytes (n=5, k=1, m=2 or 3 state) -- identical fields to the EKF's; max_iterations is comptime, not stored
Unscented KF struct    = 952 bytes (n=5, k=1, m=2 or 3 state; carries 11 cached sigma points between predict() and update())
Square-Root KF struct  = 512 bytes (n=5, k=1, m=2 or 3 state) -- identical to the EKF's: same fields (x, Q, R), just P renamed to L
Error-State KF struct  = 512 bytes (n=5, k=1, m=2 or 3 state) -- identical fields to the EKF's; inject/resetJacobian are Model decls, not stored state
process peak RSS       = ~4.4-6.1 MB (whole program: runtime + embedded dataset + all filters)
```

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
```

| RMSE | Linear KF (GPS only) | EKF, untuned Q | EKF, different Q | IEKF, untuned Q\*\*\*\* | UKF, untuned Q\*\* | SR-KF, untuned Q\*\*\* | ESKF, untuned Q\*\*\*\*\* | filterpy reference (EKF)\* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| px | 0.6342 | 0.5156 | **0.4233** | 0.5156 | 0.6856 | 0.5156 | 0.5767 | 0.5156 |
| py | 0.9290 | 0.9568 | 0.9343 | 0.9568 | 1.1894 | 0.9568 | **0.7372** | 0.9568 |
| vx | 0.8059 | 0.9840 | 1.0038 | 0.9840 | **0.7344** | 0.9840 | 0.9113 | 0.9840 |
| vy | 1.5129 | 1.3867 | 1.3689 | 1.3867 | **1.1420** | 1.3867 | 1.2419 | 1.3867 |

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
| Linear KF | Debug | ~5080-5170 | 544 |
| Linear KF | ReleaseFast | ~100 | 544 |
| Bicycle EKF (either Q) | Debug | ~4700-6440 | 320 |
| Bicycle EKF (either Q) | ReleaseFast | ~118 | 320 |
| Bicycle IEKF (3 iterations) | Debug | ~6850-7750 | 320 |
| Bicycle IEKF (3 iterations) | ReleaseFast | ~150 | 320 |
| Bicycle UKF | Debug | ~10060-12850 | 608 |
| Bicycle UKF | ReleaseFast | ~214-220 | 608 |
| Bicycle SR-KF | Debug | ~5100-5600 | 320 |
| Bicycle SR-KF | ReleaseFast | ~345-362 | 320 |
| Bicycle ESKF | Debug | ~5350-5740 | 320 |
| Bicycle ESKF | ReleaseFast | ~119-120 | 320 |

The bicycle EKF's struct is smaller than the linear filter's (320 vs. 544
bytes) despite being an EKF, because its state is the same size (`n=4`) but
it doesn't carry persistent `F`/`B`/`H` fields the way the linear filter
does — `F` is recomputed from the Jacobian each step and never stored. The
UKF's struct (608 bytes) is bigger than either: it caches all `2n+1 = 9`
sigma points between `predict()` and `update()` so `update()` doesn't need
to regenerate them from `P` a second time. The SR-KF's struct is the same
320 bytes as the EKF's (same fields, `P` just renamed to `L`). Its Debug
speed sits close to the EKF's, but ReleaseFast is ~3x the EKF's now that
`update()` runs a real Householder QR instead of a Joseph-form recursion
(same tradeoff as the IMU benchmark above — real numerical benefit, real
ReleaseFast cost). IEKF's struct is also the same 320 bytes
(`max_iterations` is comptime, not a stored field), but unlike the IMU
benchmark, its speed here is barely above the plain EKF's (~1.1-1.4x, not
the ~1.6-2.2x there) — the 3 iterations still run, but with a constant `H`
(see the accuracy table footnote above) each pass after the first is doing
real but wasted linear-algebra work on an answer that's already converged.
ESKF's struct is the same 320 bytes too (`inject`/`resetJacobian` are
`Model` decls, not stored fields) and its speed matches the plain EKF's
closely in both builds, same reasoning as the IMU benchmark above: it's the
identical recursion plus one `inject` call that's a no-op here.
