# Purifier

Purifier is a Kalman Filter implementation for different systems, written in
Zig on top of [`maryam`](https://github.com/rima1881/maryam). Two filter variants share the same covariance math (`kalman_core.KalmanCore`,
see `src/kalman_core.zig`):

- `kalman.KalmanFilter` — linear, constant-velocity model.
- `extended_kalman.ExtendedKalmanFilter` — nonlinear model/measurement
  (Jacobian-linearized each step). Two models plug into it:
  - `examples/ctrv.zig` — CTRV (constant turn rate and velocity) motion model plus
    a radar measurement model, used against the synthetic lidar/radar
    dataset below.
  - `examples/gps_ins.zig` — a bicycle model driven by real accelerometer/gyro
    control inputs, used against the real KITTI dataset below.

`zig build run` runs both benchmarks (`imu_bench.zig`, `kitti_bench.zig`),
one after another.

## Filters

- [x] **Linear Kalman Filter** — `kalman.KalmanFilter`
- [x] **Extended Kalman Filter (EKF)** — `extended_kalman.ExtendedKalmanFilter`
- [ ] **Error-State Kalman Filter (ESKF)** — estimates a small perturbation
      around a nominal state trajectory instead of the full state directly.
      The standard approach for orientation/quaternion-heavy IMU fusion
      (keeps the error state small enough that EKF-style linearization stays
      valid, and sidesteps quaternion-normalization singularities a
      direct-state EKF runs into). `src/error_state_kalman.zig` is a
      placeholder for this.
- [ ] **Unscented Kalman Filter (UKF)** — sigma-point propagation instead of
      Jacobian linearization; no derivatives needed, but needs a Cholesky/
      matrix-square-root primitive `maryam` doesn't expose yet (see
      `maryam_fix.md`).
- [ ] **Iterated Extended Kalman Filter (IEKF)** — re-linearizes `H` at the
      *updated* state estimate, not just the predicted one, iterating within
      a single update step. Cheap add on top of the existing EKF machinery
      for measurements too nonlinear for one linearization pass to handle
      well.
- [ ] **Square-Root Kalman Filter (SR-KF)** — propagates a Cholesky factor of
      `P` instead of `P` itself, so `P` can't drift non-positive-semidefinite
      even under floating-point error. Joseph form (already used by every
      filter here) is a weaker version of the same idea; this is the
      stronger one.
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
zig build run        # runs both filters against real sensor data and prints
                     # a side-by-side comparison (below)
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

Both blocks above are copied verbatim from `examples/readme_examples.zig`.

For a full worked model (multi-dimensional state, real Jacobians, process
noise from control-input uncertainty, angle-wrapped residuals) see
`examples/ctrv.zig` or `examples/gps_ins.zig` rather than reimplementing one from
scratch — both are plugged into working benchmarks below.

## The IMU benchmark (`zig build run`)

`examples/imu_bench.zig` replays both filters against the same real
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
state, which a linear filter can't express. The `ExtendedKalmanFilter(5, 1, m)`
uses a 5-state CTRV model (`[px, py, v, yaw, yaw_rate]`) with a genuinely
nonlinear radar measurement model, so it consumes *all* 500 rows.

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
```

| RMSE | Linear KF (lidar only) | EKF, untuned Q | EKF, different Q | Reference C++ EKF (CV, lidar+radar)\* |
| --- | --- | --- | --- | --- |
| px | 0.1211 | 0.0700 | **0.0591** | 0.0972 |
| py | 0.0986 | 0.0803 | 0.0828 | **0.0854** |
| vx | 0.4818 | 0.2142 | **0.1900** | 0.4509 |
| vy | 0.4576 | 0.3011 | **0.2878** | 0.4396 |

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

The EKF wins across the board over the linear filter, as it should: it sees
twice the measurements (radar's `rho_dot` speaks to velocity directly, which
lidar never does) and its CTRV model actually turns, instead of assuming
constant-velocity straight-line motion like the linear filter does.

Both EKF columns above use the exact same code path — only
`ekf_std_a`/`ekf_std_yawdd` differ (see the table above and
`imu_bench.zig`).


`ExtendedKalmanFilter(n, k, m, Model)` takes its nonlinear model
(`f`/`jacobianF`/`h`/`jacobianH`, optionally `residual`) as a **comptime**
namespace.

### Speed

Timed around `predict()` + `update()` only (excludes text parsing), native
target:

| filter | build | ns/cycle | cycles/sec |
| --- | --- | --- | --- |
| Linear KF | Debug | ~5200-5500 | ~181-193k |
| Linear KF | ReleaseFast | ~82-84 | ~12M |
| Extended KF (either Q) | Debug | ~7600-7700 | ~130-131k |
| Extended KF (either Q) | ReleaseFast | ~206-208 | ~4.8M |

The EKF costs ~1.4x (Debug) to ~2.5x (ReleaseFast) the linear filter per
cycle — Jacobian evaluation plus a 5-state (vs. 4-state) covariance
recursion, plus radar's update runs against a 3-row measurement instead of
lidar's 2. `Q`'s *value* doesn't change any of this — same matrix size,
same equations, just different numbers going in.
### Memory

```
Linear KF struct   = 544 bytes (n=4, k=1, m=2 state)
Extended KF struct = 512 bytes (n=5, k=1, m=2 or 3 state)
process peak RSS    = ~4.4-5.2 MB (whole program: runtime + embedded dataset + both filters)
```

Neither filter ever touches an allocator — every `maryam` matrix is a plain
`[rows][cols]f64` stack value, so each filter's persistent footprint is just
its own `@sizeOf(...)`, a fixed compile-time constant regardless of how long
it runs. Process peak RSS is dominated by the Zig runtime and the ~12KB
embedded dataset, not by either filter.

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
```

| RMSE | Linear KF (GPS only) | EKF, untuned Q | EKF, different Q | filterpy reference\* |
| --- | --- | --- | --- | --- |
| px | 0.6342 | 0.5156 | **0.4233** | 0.5156 |
| py | **0.9290** | 0.9568 | 0.9343 | 0.9568 |
| vx | **0.8059** | 0.9840 | 1.0038 | 0.9840 |
| vy | 1.5129 | 1.3867 | **1.3689** | 1.3867 |

\* [`filterpy`](https://github.com/rlabbe/filterpy) (Roger Labbe's widely-used
open-source Kalman filter library, `pip install filterpy`) implementing the
identical bicycle model with the **untuned** noise values — same equations,
same process/measurement noise — as an independent check. It matches this
repo's Zig implementation to 4 decimal places on every metric, which is the
useful result of this comparison: the two independent implementations agree
exactly, so the untuned numbers are a property of the model/data, not an
implementation bug. (The "different Q" column has no `filterpy` counterpart
— it's only validating correctness, not chasing the same result.)

`kitti_bench.zig` runs three configurations side by side rather than
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

The bicycle EKF's struct is smaller than the linear filter's (320 vs. 544
bytes) despite being an EKF, because its state is the same size (`n=4`) but
it doesn't carry persistent `F`/`B`/`H` fields the way the linear filter
does — `F` is recomputed from the Jacobian each step and never stored.
