# Purifier

Purifier is a Kalman Filter implementation for different systems, written in
Zig on top of [`maryam`](https://github.com/rima1881/maryam), a small
compile-time-checked linear-algebra library (matrices + a string-equation
DSL, see `src/kalman.zig`).

## Build / run / test

```sh
zig build          # compiles the library + executable
zig build test      # runs the unit tests (includes a hand-verified 1D filter case)
zig build run        # runs the filter against real sensor data and prints a report (below)
zig build run -Doptimize=ReleaseFast   # same, but built for speed
```

## The IMU benchmark (`zig build run`)

`src/imu_bench.zig` replays the filter against a real (synthetic-but-sensor-
realistic) lidar/radar dataset and scores it against ground truth.

**Data source**: [`obj_pose-laser-radar-synthetic-input.txt`](https://github.com/udacity/CarND-Extended-Kalman-Filter-Project/blob/master/data/obj_pose-laser-radar-synthetic-input.txt),
from Udacity's Extended Kalman Filter project — a bicycle-model (CTRV)
vehicle trajectory with simulated lidar (`px, py`) and radar (`rho, theta,
rho_dot`) measurements plus ground-truth `(px, py, vx, vy)` at every step.
Vendored at `src/data/laser_radar_synthetic.txt` (250 lidar rows, 250 radar
rows).

Purifier's filter (`KalmanFilter(4, 1, 2)`: state `[px, py, vx, vy]`,
constant-velocity model) is **linear**, so only the lidar rows are usable —
radar's `rho_dot` is a nonlinear function of the state and needs an EKF/UKF
to consume. Radar rows are counted but skipped.

### Accuracy

```
RMSE      px=0.1211  py=0.0986  vx=0.4818  vy=0.4576
max |err| px=0.3415  py=0.2876  vx=2.6175  vy=1.2662
```

| | this filter (lidar only) | reference EKF (lidar + radar)* |
| --- | --- | --- |
| position RMSE (px, py) | 0.12, 0.10 | ~0.09–0.10 |
| velocity RMSE (vx, vy) | 0.48, 0.46 | ~0.40, 0.30 |

\* approximate published numbers for the full lidar+radar Extended Kalman
Filter from the same Udacity project.

Position tracking is essentially on par with the full sensor-fusion
reference. Velocity is noticeably worse, for two structural reasons rather
than a bug:

1. **Velocity isn't measured, only inferred.** Lidar gives `(px, py)`
   directly every step; velocity is corrected only indirectly through the
   position/velocity cross-covariance built up in `P`.
2. **Model mismatch during turns.** The ground truth follows a
   constant-turn-rate path; this filter assumes constant velocity in a
   straight line and never rotates the velocity vector, so error concentrates
   right around turns (hence `max|err|` on velocity being much larger,
   relatively, than its RMSE).
3. Radar's `rho_dot` — the one measurement that speaks to velocity directly —
   isn't used at all, since it's nonlinear.

See `maryam_fix.md` for possible library-level changes (e.g. a Cholesky
solve instead of full matrix inversion for the Kalman gain) that would help
here without needing an EKF.

### Speed

Timed around `predict()` + `update()` only (excludes text parsing), native
target:

| build | ns/cycle | cycles/sec | real-time headroom |
| --- | --- | --- | --- |
| Debug | ~5200 | ~192k | ~19,000x |
| ReleaseFast | ~82 | ~12.2M | ~1,200,000x |

"Real-time headroom" is relative to the dataset's own lidar rate (~100ms
between lidar samples, i.e. 10 Hz) — even in an unoptimized Debug build, a
single predict+update cycle for this 4-state filter takes microseconds, far
under the sensor's own sampling interval.

### Memory

```
filter struct = 544 bytes (n=4, k=1, m=2 state)
process peak RSS = ~4.4-5.1 MB (whole program: runtime + embedded dataset + filter)
```

`kalman.zig` never touches an allocator — every `maryam` matrix is a plain
`[rows][cols]f64` stack value, so the entire filter's persistent footprint is
just `@sizeOf(KalmanFilter(n, k, m))` (544 bytes for this 4-state/2-measurement
case, and it's a fixed compile-time constant regardless of how long the
filter runs — no leaks possible, nothing to `free`). Process peak RSS is
dominated by the Zig runtime and the ~12KB embedded dataset, not by the
filter itself.
