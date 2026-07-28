//! Runs the filters against the same real lidar/radar dataset (synthetic but
//! sensor-realistic) and reports RMSE against ground truth, plus speed, for
//! each -- so they can be compared directly. Dataset source:
//! https://github.com/udacity/CarND-Extended-Kalman-Filter-Project
//!
//! `kalman.KalmanFilter` is linear (constant-velocity model), so it can only
//! consume the "L" (lidar, direct x/y position) rows; "R" (radar, polar
//! rho/theta/rho_dot) rows require a nonlinear observation model and are
//! skipped entirely.
//!
//! `extended_kalman.ExtendedKalmanFilter`,
//! `iterated_extended_kalman.IteratedExtendedKalmanFilter`,
//! `unscented_kalman.UnscentedKalmanFilter`,
//! `square_root_kalman.SquareRootKalmanFilter`, and
//! `error_state_kalman.ErrorStateKalmanFilter` all use a CTRV (constant turn
//! rate and velocity) model (see `ctrv.zig`) with a genuinely nonlinear
//! radar measurement model, so all five consume *all* 500 rows.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Purifier = @import("Purifier");
const kalman = Purifier.kalman;
const extended_kalman = Purifier.extended_kalman;
const iterated_extended_kalman = Purifier.iterated_extended_kalman;
const unscented_kalman = Purifier.unscented_kalman;
const square_root_kalman = Purifier.square_root_kalman;
const error_state_kalman = Purifier.error_state_kalman;
const ctrv = @import("ctrv.zig");
const cv = @import("constant_velocity.zig");
const maryam = @import("maryam");
const bench_common = @import("bench_common.zig");
const BenchResult = bench_common.BenchResult;
const ErrorAccumulator = bench_common.ErrorAccumulator;
const report = bench_common.report;

const data = @embedFile("data/laser_radar_synthetic.txt");

// ---- Linear KF: constant-velocity model, lidar only ----

const KF = kalman.KalmanFilter(4, 1, 2);

fn runLinear(io: Io) !BenchResult {
    var filter: KF = undefined;
    var initialized = false;
    var prev_t: i64 = 0;
    var acc = ErrorAccumulator{};

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
                    var m = cv.StateVec.zero();
                    m.data = .{ .{px}, .{py}, .{0}, .{0} };
                    break :blk m;
                },
                .P = blk: {
                    var m = cv.StateMat.zero();
                    m.data[0][0] = 1;
                    m.data[1][1] = 1;
                    m.data[2][2] = 1000;
                    m.data[3][3] = 1000;
                    break :blk m;
                },
                .F = cv.identity(),
                .B = cv.ControlMat.zero(),
                .Q = cv.StateMat.zero(),
                .H = cv.measureH(),
                .R = blk: {
                    var m = cv.MeasureNoise.zero();
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

        filter.F = cv.stateTransition(dt);
        filter.Q = cv.processNoise(dt, 9.0, 9.0);

        const t0 = Io.Timestamp.now(io, .awake);
        filter.predict(cv.ControlVec.zero());

        var z = cv.MeasureVec.zero();
        z.data = .{ .{px}, .{py} };
        const update_result = filter.update(z);
        const t1 = Io.Timestamp.now(io, .awake);
        acc.filter_ns += t0.durationTo(t1).nanoseconds;
        update_result catch {
            acc.recordSingular();
            continue;
        };

        acc.record(.{
            filter.x.data[0][0] - gt_px,
            filter.x.data[1][0] - gt_py,
            filter.x.data[2][0] - gt_vx,
            filter.x.data[3][0] - gt_vy,
        });
    }

    return acc.finish(@sizeOf(KF));
}

// ---- EKF/UKF/SR-KF: CTRV model, lidar + radar ----

const LidarEKF = extended_kalman.ExtendedKalmanFilter(5, 1, 2, ctrv.LidarModel);
const RadarEKF = extended_kalman.ExtendedKalmanFilter(5, 1, 3, ctrv.RadarModel);

// Reuses ctrv.LidarModel/ctrv.RadarModel unchanged for the UKF and SR-KF:
// both only ever call .f/.h/.jacobianF/.jacobianH as needed and never
// require anything ctrv.zig doesn't already provide for the EKF's sake.
const LidarUKF = unscented_kalman.UnscentedKalmanFilter(5, 1, 2, ctrv.LidarModel);
const RadarUKF = unscented_kalman.UnscentedKalmanFilter(5, 1, 3, ctrv.RadarModel);

const LidarSRKF = square_root_kalman.SquareRootKalmanFilter(5, 1, 2, ctrv.LidarModel);
const RadarSRKF = square_root_kalman.SquareRootKalmanFilter(5, 1, 3, ctrv.RadarModel);

// Reuses ctrv.LidarModel/ctrv.RadarModel unchanged too: they now also
// supply `inject`/`resetJacobian` (see ctrv.zig), which only
// ErrorStateKalmanFilter consumes -- the EKF/IEKF/UKF/SR-KF instances above
// never look those decls up, so adding them didn't change anything about
// this file's other five filter types.
const LidarESKF = error_state_kalman.ErrorStateKalmanFilter(5, 1, 2, ctrv.LidarModel);
const RadarESKF = error_state_kalman.ErrorStateKalmanFilter(5, 1, 3, ctrv.RadarModel);

// max_iterations = 3, chosen empirically: swept 1/2/3/4/5/8/10 on this
// dataset and the result is *not* monotonic (a known property of
// undamped/vanilla Gauss-Newton IEKF -- with no step damping or line
// search, a linearization can overshoot and the next iteration overcorrects
// rather than settling). 3 iterations is the best performer of the sweep
// (beats the plain EKF on vx/vy), 4/5/8 are all worse than even a single
// pass -- see Readme.md's IMU benchmark section for the full sweep and the
// discussion of why more iterations isn't reliably better here.
const LidarIEKF = iterated_extended_kalman.IteratedExtendedKalmanFilter(5, 1, 2, ctrv.LidarModel, 3);
const RadarIEKF = iterated_extended_kalman.IteratedExtendedKalmanFilter(5, 1, 3, ctrv.RadarModel, 3);

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

// Shared by the EKF, IEKF, UKF, SR-KF, and ESKF configurations below: only
// the filter types (`Lidar`/`Radar`) differ between them, everything else --
// dataset parsing, initialization, dt/Q computation, error accumulation --
// is identical. `bench_common.predictErr` absorbs one real behavioral
// difference (EKF's, IEKF's, and ESKF's predict() can't fail; UKF's and
// SR-KF's can, since both run a Cholesky decomposition), and
// `bench_common.makeFilter`/`covOf` absorb another (SR-KF's
// covariance-representation field is named `L`, not `P`, since it stores a
// Cholesky factor rather than P itself). IEKF and ESKF need no changes here
// at all -- both have `predict()`/`update()` signatures identical to the
// plain EKF's.
fn runFilterPair(comptime Lidar: type, comptime Radar: type, io: Io, std_a: f64, std_yawdd: f64) !BenchResult {
    var x = ctrv.StateVec.zero();
    var cov = maryam.I(5);
    var initialized = false;
    var prev_t: i64 = 0;
    var acc = ErrorAccumulator{};

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
            // (P = I, i.e. cov = I -- I is trivially its own Cholesky factor,
            // so this same value is correct whether `cov` ends up meaning P
            // or L) and the filter has to converge onto them from motion.
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
            var filt = bench_common.makeFilter(Radar, x, cov, Q, ekf_R_radar);
            update_err = bench_common.predictErr(&filt, dt_vec);
            if (update_err == null) {
                var z = ctrv.RadarVec.zero();
                z.data = .{ .{meas1}, .{meas2}, .{meas3} };
                filt.update(z) catch |e| {
                    update_err = e;
                };
            }
            x = filt.x;
            cov = bench_common.covOf(ctrv.StateMat, filt);
        } else {
            var filt = bench_common.makeFilter(Lidar, x, cov, Q, ekf_R_lidar);
            update_err = bench_common.predictErr(&filt, dt_vec);
            if (update_err == null) {
                var z = ctrv.LidarVec.zero();
                z.data = .{ .{meas1}, .{meas2} };
                filt.update(z) catch |e| {
                    update_err = e;
                };
            }
            x = filt.x;
            cov = bench_common.covOf(ctrv.StateMat, filt);
        }
        const t1 = Io.Timestamp.now(io, .awake);
        acc.filter_ns += t0.durationTo(t1).nanoseconds;

        if (update_err != null) {
            acc.recordSingular();
            continue;
        }

        const est_vx = x.data[2][0] * @cos(x.data[3][0]);
        const est_vy = x.data[2][0] * @sin(x.data[3][0]);
        acc.record(.{
            x.data[0][0] - gt_px,
            x.data[1][0] - gt_py,
            est_vx - gt_vx,
            est_vy - gt_vy,
        });
    }

    return acc.finish(@max(@sizeOf(Lidar), @sizeOf(Radar)));
}

pub fn run(w: *Io.Writer, io: Io) !void {
    const linear = try runLinear(io);
    const ekf_untuned = try runFilterPair(LidarEKF, RadarEKF, io, ekf_std_a_untuned, ekf_std_yawdd_untuned);
    const ekf_alt = try runFilterPair(LidarEKF, RadarEKF, io, ekf_std_a_alt, ekf_std_yawdd_alt);
    // Same untuned Q as "EKF, untuned Q" for both -- IEKF and SR-KF are
    // both built directly on top of the plain EKF (re-linearize more, or
    // represent P differently), so holding Q fixed isolates exactly what
    // each one changes.
    const iekf = try runFilterPair(LidarIEKF, RadarIEKF, io, ekf_std_a_untuned, ekf_std_yawdd_untuned);
    const ukf = try runFilterPair(LidarUKF, RadarUKF, io, ekf_std_a_untuned, ekf_std_yawdd_untuned);
    const srkf = try runFilterPair(LidarSRKF, RadarSRKF, io, ekf_std_a_untuned, ekf_std_yawdd_untuned);
    // Same untuned Q as the plain EKF, same reasoning as IEKF/UKF/SR-KF
    // above: isolates exactly what ErrorStateKalmanFilter changes (state
    // composition through ctrv.inject) rather than also varying Q.
    const eskf = try runFilterPair(LidarESKF, RadarESKF, io, ekf_std_a_untuned, ekf_std_yawdd_untuned);

    try w.print("dataset: 250 lidar rows, 250 radar rows (all filters see the same data)\n\n", .{});
    try report(w, "Linear KF -- constant-velocity model, lidar only", linear);
    try report(w, "Extended KF (untuned Q) -- CTRV model, lidar + radar", ekf_untuned);
    try report(w, "Extended KF (different Q) -- CTRV model, lidar + radar", ekf_alt);
    try report(w, "Iterated EKF, max 3 iterations (same Q as untuned EKF) -- CTRV model, lidar + radar", iekf);
    try report(w, "Unscented KF (same Q as untuned EKF) -- CTRV model, lidar + radar", ukf);
    try report(w, "Square-Root KF (same Q as untuned EKF) -- CTRV model, lidar + radar", srkf);
    try report(w, "Error-State KF (same Q as untuned EKF) -- CTRV model, lidar + radar", eskf);

    try w.print("-- comparison: linear -> EKF untuned -> EKF different Q -> IEKF -> UKF -> SR-KF -> ESKF --\n", .{});
    try w.print("RMSE px  {d:.4} -> {d:.4} -> {d:.4} -> {d:.4} -> {d:.4} -> {d:.4} -> {d:.4}\n", .{ linear.rmse[0], ekf_untuned.rmse[0], ekf_alt.rmse[0], iekf.rmse[0], ukf.rmse[0], srkf.rmse[0], eskf.rmse[0] });
    try w.print("RMSE py  {d:.4} -> {d:.4} -> {d:.4} -> {d:.4} -> {d:.4} -> {d:.4} -> {d:.4}\n", .{ linear.rmse[1], ekf_untuned.rmse[1], ekf_alt.rmse[1], iekf.rmse[1], ukf.rmse[1], srkf.rmse[1], eskf.rmse[1] });
    try w.print("RMSE vx  {d:.4} -> {d:.4} -> {d:.4} -> {d:.4} -> {d:.4} -> {d:.4} -> {d:.4}\n", .{ linear.rmse[2], ekf_untuned.rmse[2], ekf_alt.rmse[2], iekf.rmse[2], ukf.rmse[2], srkf.rmse[2], eskf.rmse[2] });
    try w.print("RMSE vy  {d:.4} -> {d:.4} -> {d:.4} -> {d:.4} -> {d:.4} -> {d:.4} -> {d:.4}\n", .{ linear.rmse[3], ekf_untuned.rmse[3], ekf_alt.rmse[3], iekf.rmse[3], ukf.rmse[3], srkf.rmse[3], eskf.rmse[3] });
    try w.print("speed    {d:.0} ns/cycle -> {d:.0} ns/cycle -> {d:.0} ns/cycle -> {d:.0} ns/cycle -> {d:.0} ns/cycle -> {d:.0} ns/cycle -> {d:.0} ns/cycle\n\n", .{
        linear.avg_ns_per_cycle, ekf_untuned.avg_ns_per_cycle, ekf_alt.avg_ns_per_cycle, iekf.avg_ns_per_cycle, ukf.avg_ns_per_cycle, srkf.avg_ns_per_cycle, eskf.avg_ns_per_cycle,
    });

    if (builtin.os.tag != .windows) {
        const ru = std.posix.getrusage(std.posix.rusage.SELF);
        const peak_kb: i64 = if (builtin.os.tag == .macos) @divTrunc(ru.maxrss, 1024) else ru.maxrss;
        try w.print("process peak RSS = {d} KB (whole program: runtime + embedded dataset + all filters)\n", .{peak_kb});
    }

    try w.flush();
}
