#!/usr/bin/env python3
"""
Builds a longer synthetic lidar/radar CTRV dataset, in the exact same format
`imu_bench.zig` already parses (see `data/laser_radar_synthetic.txt`, the
Udacity CTRV EKF project's dataset) -- generated rather than downloaded,
specifically to test whether `AdaptiveKalmanFilter`'s online Q-estimation
needs more data than that 500-row dataset (or the 154-frame KITTI one) gives
it to converge on a useful estimate. See Readme.md's Adaptive Kalman Filter
findings for what this was used to check and what it found.

Vehicle follows a fixed CTRV (constant turn rate and velocity) trajectory --
alternating straight legs and turns, at varying speed -- not a single
constant-radius circle, so the process noise actually has something
non-trivial to explain (same reasoning the original dataset's own S-shaped
path has). Lidar (direct px/py) and radar (rho/theta/rho_dot) readings
alternate row-by-row like the original, with Gaussian noise matching the
R matrices `imu_bench.zig` already assumes (lidar std=0.15, radar
rho/theta/rho_dot std=0.3/0.03/0.3).

Output columns (tab-separated), one row per measurement:
  L  px       py                t_micros  gt_px  gt_py  gt_vx  gt_vy
  R  rho  theta  rho_dot         t_micros  gt_px  gt_py  gt_vx  gt_vy
"""
import math
import random

OUT = "../data/ctrv_long_synthetic.txt"
SEED = 42
DT = 0.05  # seconds between measurements, matching the original dataset
N_STEPS = 5000  # 2500 lidar + 2500 radar rows -- 10x the original dataset

LIDAR_NOISE_STD = 0.15  # matches ekf_R_lidar's 0.0225 variance
RADAR_RHO_STD = 0.3
RADAR_THETA_STD = 0.03
RADAR_RHO_DOT_STD = 0.3


def main():
    rng = random.Random(SEED)

    px, py, v, yaw, yawd = 0.0, 0.0, 3.0, 0.0, 0.0

    # A fixed sequence of (duration_steps, target_v, target_yawd) legs,
    # cycled, so the vehicle actually turns and changes speed repeatedly
    # over the run instead of settling into one constant-radius circle.
    legs = [
        (200, 4.0, 0.0),
        (100, 4.0, 0.6),
        (200, 6.0, 0.0),
        (100, 6.0, -0.5),
        (150, 3.0, 0.0),
        (100, 3.0, 0.8),
        (200, 5.0, 0.0),
        (100, 5.0, -0.7),
    ]

    out = []
    t_micros = 0
    leg_idx = 0
    leg_remaining, target_v, target_yawd = legs[0]

    for i in range(N_STEPS):
        if leg_remaining <= 0:
            leg_idx = (leg_idx + 1) % len(legs)
            leg_remaining, target_v, target_yawd = legs[leg_idx]
        leg_remaining -= 1

        # Ease v/yawd toward this leg's targets rather than snapping, so F's
        # own constant-velocity/turn-rate assumption is only ever slightly
        # wrong within a step (matching how a real vehicle actually
        # accelerates/turns), not a step-function violation of the model.
        v += 0.02 * (target_v - v)
        yawd += 0.05 * (target_yawd - yawd)

        if abs(yawd) > 1e-4:
            px += (v / yawd) * (math.sin(yaw + yawd * DT) - math.sin(yaw))
            py += (v / yawd) * (-math.cos(yaw + yawd * DT) + math.cos(yaw))
        else:
            px += v * math.cos(yaw) * DT
            py += v * math.sin(yaw) * DT
        yaw += yawd * DT

        gt_vx = v * math.cos(yaw)
        gt_vy = v * math.sin(yaw)

        t_micros += int(DT * 1_000_000)

        if i % 2 == 0:
            mx = px + rng.gauss(0, LIDAR_NOISE_STD)
            my = py + rng.gauss(0, LIDAR_NOISE_STD)
            out.append(f"L\t{mx:.6e}\t{my:.6e}\t{t_micros}\t{px:.6e}\t{py:.6e}\t{gt_vx:.6e}\t{gt_vy:.6e}")
        else:
            rho = math.hypot(px, py)
            theta = math.atan2(py, px)
            rho_dot = (px * gt_vx + py * gt_vy) / rho if rho > 1e-6 else 0.0
            mrho = rho + rng.gauss(0, RADAR_RHO_STD)
            mtheta = theta + rng.gauss(0, RADAR_THETA_STD)
            mrho_dot = rho_dot + rng.gauss(0, RADAR_RHO_DOT_STD)
            out.append(f"R\t{mrho:.6e}\t{mtheta:.6e}\t{mrho_dot:.6e}\t{t_micros}\t{px:.6e}\t{py:.6e}\t{gt_vx:.6e}\t{gt_vy:.6e}")

    with open(OUT, "w") as f:
        f.write("\n".join(out) + "\n")

    print(f"wrote {N_STEPS} rows to {OUT}")


if __name__ == "__main__":
    main()
