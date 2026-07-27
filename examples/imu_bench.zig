//! Runs two filters against the same real lidar/radar dataset (synthetic but
//! sensor-realistic) and reports RMSE against ground truth, plus speed, for
//! each -- so they can be compared directly. Dataset source:
//! https://github.com/udacity/CarND-Extended-Kalman-Filter-Project
//!
//! `kalman.KalmanFilter` is linear (constant-velocity model), so it can only
//! consume the "L" (lidar, direct x/y position) rows; "R" (radar, polar
//! rho/theta/rho_dot) rows require a nonlinear observation model and are
//! skipped entirely.
//!
//! `extended_kalman.ExtendedKalmanFilter` uses a CTRV (constant turn rate and
//! velocity) model (see `ctrv.zig`) with a genuinely nonlinear radar
//! measurement model, so it consumes *all* 500 rows.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Purifier = @import("Purifier");
const kalman = Purifier.kalman;
const extended_kalman = Purifier.extended_kalman;
const ctrv = @import("ctrv.zig");
const maryam = @import("maryam");
const bench_common = @import("bench_common.zig");
const BenchResult = bench_common.BenchResult;
const report = bench_common.report;

const data = @embedFile("data/laser_radar_synthetic.txt");

// ---- Linear KF: constant-velocity model, lidar only ----

const KF = kalman.KalmanFilter(4, 1, 2);
const StateVec = maryam.MatrixType(4, 1);
const StateMat = maryam.MatrixType(4, 4);
const ControlVec = maryam.MatrixType(1, 1);
const ControlMat = maryam.MatrixType(4, 1);
const MeasureVec = maryam.MatrixType(2, 1);
const MeasureMat = maryam.MatrixType(2, 4);
const MeasureNoise = maryam.MatrixType(2, 2);

fn identityStateMat() StateMat {
    var m = StateMat.zero();
    m.data[0][0] = 1;
    m.data[1][1] = 1;
    m.data[2][2] = 1;
    m.data[3][3] = 1;
    return m;
}

// Constant-velocity state transition: px += vx*dt, py += vy*dt.
fn stateTransition(dt: f64) StateMat {
    var m = identityStateMat();
    m.data[0][2] = dt;
    m.data[1][3] = dt;
    return m;
}

// Standard discretized-white-noise-acceleration process noise.
fn linearProcessNoise(dt: f64, ax: f64, ay: f64) StateMat {
    const dt2 = dt * dt;
    const dt3 = dt2 * dt;
    const dt4 = dt3 * dt;
    var m = StateMat.zero();
    m.data[0][0] = dt4 / 4 * ax;
    m.data[0][2] = dt3 / 2 * ax;
    m.data[2][0] = dt3 / 2 * ax;
    m.data[2][2] = dt2 * ax;
    m.data[1][1] = dt4 / 4 * ay;
    m.data[1][3] = dt3 / 2 * ay;
    m.data[3][1] = dt3 / 2 * ay;
    m.data[3][3] = dt2 * ay;
    return m;
}

fn runLinear(io: Io) !BenchResult {
    var filter: KF = undefined;
    var initialized = false;
    var prev_t: i64 = 0;

    var sum_sq = [4]f64{ 0, 0, 0, 0 };
    var max_abs = [4]f64{ 0, 0, 0, 0 };
    var n_scored: usize = 0;
    var n_singular: usize = 0;
    var filter_ns: i96 = 0;

    var lines = std.mem.tokenizeScalar(u8, data, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trimEnd(u8, line_raw, "\r");
        if (line.len == 0) continue;

        var fields = std.mem.tokenizeScalar(u8, line, '\t');
        const kind = fields.next().?;
        if (!std.mem.eql(u8, kind, "L")) continue;

        const px = try std.fmt.parseFloat(f64, fields.next().?);
        const py = try std.fmt.parseFloat(f64, fields.next().?);
        const t = try std.fmt.parseInt(i64, fields.next().?, 10);
        const gt_px = try std.fmt.parseFloat(f64, fields.next().?);
        const gt_py = try std.fmt.parseFloat(f64, fields.next().?);
        const gt_vx = try std.fmt.parseFloat(f64, fields.next().?);
        const gt_vy = try std.fmt.parseFloat(f64, fields.next().?);

        if (!initialized) {
            filter = .{
                .x = blk: {
                    var m = StateVec.zero();
                    m.data = .{ .{px}, .{py}, .{0}, .{0} };
                    break :blk m;
                },
                .P = blk: {
                    var m = StateMat.zero();
                    m.data[0][0] = 1;
                    m.data[1][1] = 1;
                    m.data[2][2] = 1000;
                    m.data[3][3] = 1000;
                    break :blk m;
                },
                .F = identityStateMat(),
                .B = ControlMat.zero(),
                .Q = StateMat.zero(),
                .H = blk: {
                    var m = MeasureMat.zero();
                    m.data[0][0] = 1;
                    m.data[1][1] = 1;
                    break :blk m;
                },
                .R = blk: {
                    var m = MeasureNoise.zero();
                    m.data[0][0] = 0.0225;
                    m.data[1][1] = 0.0225;
                    break :blk m;
                },
            };
            prev_t = t;
            initialized = true;
            continue;
        }

        const dt: f64 = @as(f64, @floatFromInt(t - prev_t)) / 1_000_000.0;
        prev_t = t;

        filter.F = stateTransition(dt);
        filter.Q = linearProcessNoise(dt, 9.0, 9.0);

        const t0 = Io.Timestamp.now(io, .awake);
        filter.predict(ControlVec.zero());

        var z = MeasureVec.zero();
        z.data = .{ .{px}, .{py} };
        const update_result = filter.update(z);
        const t1 = Io.Timestamp.now(io, .awake);
        filter_ns += t0.durationTo(t1).nanoseconds;
        update_result catch {
            n_singular += 1;
            continue;
        };

        const errs = [4]f64{
            filter.x.data[0][0] - gt_px,
            filter.x.data[1][0] - gt_py,
            filter.x.data[2][0] - gt_vx,
            filter.x.data[3][0] - gt_vy,
        };
        for (0..4) |i| {
            sum_sq[i] += errs[i] * errs[i];
            max_abs[i] = @max(max_abs[i], @abs(errs[i]));
        }
        n_scored += 1;
    }

    const n: f64 = @floatFromInt(n_scored);
    return .{
        .n_scored = n_scored,
        .n_singular = n_singular,
        .rmse = .{ @sqrt(sum_sq[0] / n), @sqrt(sum_sq[1] / n), @sqrt(sum_sq[2] / n), @sqrt(sum_sq[3] / n) },
        .max_abs = max_abs,
        .avg_ns_per_cycle = @as(f64, @floatFromInt(filter_ns)) / n,
        .struct_bytes = @sizeOf(KF),
    };
}

// ---- EKF: CTRV model, lidar + radar ----

const LidarEKF = extended_kalman.ExtendedKalmanFilter(5, 1, 2, ctrv.LidarModel);
const RadarEKF = extended_kalman.ExtendedKalmanFilter(5, 1, 3, ctrv.RadarModel);

// Untuned defaults, not fit to this dataset: std_a is longitudinal
// acceleration noise (m/s^2), std_yawdd is yaw-acceleration noise (rad/s^2).
const ekf_std_a_untuned = 2.0;
const ekf_std_yawdd_untuned = 0.5;

// A different Q, found with a coarse grid search over this exact dataset
// (see git history for the sweep) minimizing summed RMSE across
// px/py/vx/vy -- kept alongside the untuned defaults above rather than
// replacing them, so the effect of changing Q is visible instead of hidden.
const ekf_std_a_alt = 0.5;
const ekf_std_yawdd_alt = 0.5;

// Same lidar noise assumption as the linear filter, for a fair comparison.
const ekf_R_lidar: ctrv.LidarNoise = blk: {
    var m = ctrv.LidarNoise.zero();
    m.data[0][0] = 0.0225;
    m.data[1][1] = 0.0225;
    break :blk m;
};

// Standard published radar noise stddevs for this dataset: rho 0.3m,
// bearing 0.03rad, range-rate 0.3m/s.
const ekf_R_radar: ctrv.RadarNoise = blk: {
    var m = ctrv.RadarNoise.zero();
    m.data[0][0] = 0.09;
    m.data[1][1] = 0.0009;
    m.data[2][2] = 0.09;
    break :blk m;
};

fn runEkf(io: Io, std_a: f64, std_yawdd: f64) !BenchResult {
    var x = ctrv.StateVec.zero();
    var P = maryam.I(5);
    var initialized = false;
    var prev_t: i64 = 0;

    var sum_sq = [4]f64{ 0, 0, 0, 0 };
    var max_abs = [4]f64{ 0, 0, 0, 0 };
    var n_scored: usize = 0;
    var n_singular: usize = 0;
    var filter_ns: i96 = 0;

    var lines = std.mem.tokenizeScalar(u8, data, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trimEnd(u8, line_raw, "\r");
        if (line.len == 0) continue;

        var fields = std.mem.tokenizeScalar(u8, line, '\t');
        const kind = fields.next().?;
        const is_radar = std.mem.eql(u8, kind, "R");
        if (!is_radar and !std.mem.eql(u8, kind, "L")) continue;

        const meas1 = try std.fmt.parseFloat(f64, fields.next().?);
        const meas2 = try std.fmt.parseFloat(f64, fields.next().?);
        const meas3: f64 = if (is_radar) try std.fmt.parseFloat(f64, fields.next().?) else 0;
        const t = try std.fmt.parseInt(i64, fields.next().?, 10);
        const gt_px = try std.fmt.parseFloat(f64, fields.next().?);
        const gt_py = try std.fmt.parseFloat(f64, fields.next().?);
        const gt_vx = try std.fmt.parseFloat(f64, fields.next().?);
        const gt_vy = try std.fmt.parseFloat(f64, fields.next().?);

        if (!initialized) {
            // v, yaw, yaw_rate can't be observed from a single measurement of
            // either sensor, so they start at 0 with wide-open uncertainty
            // (P = I) and the filter has to converge onto them from motion.
            if (is_radar) {
                x.data[0][0] = meas1 * @cos(meas2); // rho * cos(theta)
                x.data[1][0] = meas1 * @sin(meas2); // rho * sin(theta)
            } else {
                x.data[0][0] = meas1;
                x.data[1][0] = meas2;
            }
            prev_t = t;
            initialized = true;
            continue;
        }

        const dt: f64 = @as(f64, @floatFromInt(t - prev_t)) / 1_000_000.0;
        prev_t = t;

        const Q = ctrv.processNoise(x, dt, std_a, std_yawdd);
        const dt_vec: ctrv.DtVec = .{ .data = .{.{dt}} };

        const t0 = Io.Timestamp.now(io, .awake);
        var update_err: ?maryam.EvalError = null;
        if (is_radar) {
            var filt = RadarEKF{
                .x = x,
                .P = P,
                .Q = Q,
                .R = ekf_R_radar,
            };
            filt.predict(dt_vec);
            var z = ctrv.RadarVec.zero();
            z.data = .{ .{meas1}, .{meas2}, .{meas3} };
            filt.update(z) catch |e| {
                update_err = e;
            };
            x = filt.x;
            P = filt.P;
        } else {
            var filt = LidarEKF{
                .x = x,
                .P = P,
                .Q = Q,
                .R = ekf_R_lidar,
            };
            filt.predict(dt_vec);
            var z = ctrv.LidarVec.zero();
            z.data = .{ .{meas1}, .{meas2} };
            filt.update(z) catch |e| {
                update_err = e;
            };
            x = filt.x;
            P = filt.P;
        }
        const t1 = Io.Timestamp.now(io, .awake);
        filter_ns += t0.durationTo(t1).nanoseconds;

        if (update_err != null) {
            n_singular += 1;
            continue;
        }

        const est_vx = x.data[2][0] * @cos(x.data[3][0]);
        const est_vy = x.data[2][0] * @sin(x.data[3][0]);
        const errs = [4]f64{
            x.data[0][0] - gt_px,
            x.data[1][0] - gt_py,
            est_vx - gt_vx,
            est_vy - gt_vy,
        };
        for (0..4) |i| {
            sum_sq[i] += errs[i] * errs[i];
            max_abs[i] = @max(max_abs[i], @abs(errs[i]));
        }
        n_scored += 1;
    }

    const n: f64 = @floatFromInt(n_scored);
    return .{
        .n_scored = n_scored,
        .n_singular = n_singular,
        .rmse = .{ @sqrt(sum_sq[0] / n), @sqrt(sum_sq[1] / n), @sqrt(sum_sq[2] / n), @sqrt(sum_sq[3] / n) },
        .max_abs = max_abs,
        .avg_ns_per_cycle = @as(f64, @floatFromInt(filter_ns)) / n,
        .struct_bytes = @max(@sizeOf(LidarEKF), @sizeOf(RadarEKF)),
    };
}

pub fn run(w: *Io.Writer, io: Io) !void {
    const linear = try runLinear(io);
    const ekf_untuned = try runEkf(io, ekf_std_a_untuned, ekf_std_yawdd_untuned);
    const ekf_alt = try runEkf(io, ekf_std_a_alt, ekf_std_yawdd_alt);

    try w.print("dataset: 250 lidar rows, 250 radar rows (all filters see the same data)\n\n", .{});
    try report(w, "Linear KF -- constant-velocity model, lidar only", linear);
    try report(w, "Extended KF (untuned Q) -- CTRV model, lidar + radar", ekf_untuned);
    try report(w, "Extended KF (different Q) -- CTRV model, lidar + radar", ekf_alt);

    try w.print("-- comparison: linear -> EKF untuned -> EKF different Q --\n", .{});
    try w.print("RMSE px  {d:.4} -> {d:.4} -> {d:.4}\n", .{ linear.rmse[0], ekf_untuned.rmse[0], ekf_alt.rmse[0] });
    try w.print("RMSE py  {d:.4} -> {d:.4} -> {d:.4}\n", .{ linear.rmse[1], ekf_untuned.rmse[1], ekf_alt.rmse[1] });
    try w.print("RMSE vx  {d:.4} -> {d:.4} -> {d:.4}\n", .{ linear.rmse[2], ekf_untuned.rmse[2], ekf_alt.rmse[2] });
    try w.print("RMSE vy  {d:.4} -> {d:.4} -> {d:.4}\n", .{ linear.rmse[3], ekf_untuned.rmse[3], ekf_alt.rmse[3] });
    try w.print("speed    {d:.0} ns/cycle -> {d:.0} ns/cycle -> {d:.0} ns/cycle\n\n", .{
        linear.avg_ns_per_cycle, ekf_untuned.avg_ns_per_cycle, ekf_alt.avg_ns_per_cycle,
    });

    if (builtin.os.tag != .windows) {
        const ru = std.posix.getrusage(std.posix.rusage.SELF);
        const peak_kb: i64 = if (builtin.os.tag == .macos) @divTrunc(ru.maxrss, 1024) else ru.maxrss;
        try w.print("process peak RSS = {d} KB (whole program: runtime + embedded dataset + all filters)\n", .{peak_kb});
    }

    try w.flush();
}
