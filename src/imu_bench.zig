//! Runs the Kalman filter against a real lidar/radar dataset (synthetic but
//! sensor-realistic) and reports RMSE against ground truth. Dataset source:
//! https://github.com/udacity/CarND-Extended-Kalman-Filter-Project
//!
//! The filter here is linear, so only the "L" (lidar, direct x/y position)
//! measurements are usable; "R" (radar, polar rho/theta/rho_dot) rows require
//! a nonlinear observation model and are skipped.

const std = @import("std");
const Io = std.Io;
const Purifier = @import("Purifier");
const maryam = @import("maryam");

const KF = Purifier.kalman.KalmanFilter(4, 1, 2);
const StateVec = maryam.MatrixType(4, 1);
const StateMat = maryam.MatrixType(4, 4);
const ControlVec = maryam.MatrixType(1, 1);
const ControlMat = maryam.MatrixType(4, 1);
const MeasureVec = maryam.MatrixType(2, 1);
const MeasureMat = maryam.MatrixType(2, 4);
const MeasureNoise = maryam.MatrixType(2, 2);

const data = @embedFile("data/laser_radar_synthetic.txt");

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
fn processNoise(dt: f64, ax: f64, ay: f64) StateMat {
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

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const w = &stdout_file_writer.interface;

    var filter: KF = undefined;
    var initialized = false;
    var prev_t: i64 = 0;

    var sum_sq = [4]f64{ 0, 0, 0, 0 };
    var max_abs = [4]f64{ 0, 0, 0, 0 };
    var n_laser: usize = 0;
    var n_radar: usize = 0;
    var n_scored: usize = 0;
    var n_singular: usize = 0;

    var lines = std.mem.tokenizeScalar(u8, data, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trimEnd(u8, line_raw, "\r");
        if (line.len == 0) continue;

        var fields = std.mem.tokenizeScalar(u8, line, '\t');
        const kind = fields.next().?;

        if (std.mem.eql(u8, kind, "R")) {
            n_radar += 1;
            continue;
        }
        if (!std.mem.eql(u8, kind, "L")) continue;
        n_laser += 1;

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
        filter.Q = processNoise(dt, 9.0, 9.0);
        filter.predict(ControlVec.zero());

        var z = MeasureVec.zero();
        z.data = .{ .{px}, .{py} };
        filter.update(z) catch {
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
    try w.print("dataset: {d} lidar rows, {d} radar rows (radar skipped: linear filter can't consume polar measurements)\n", .{ n_laser, n_radar });
    try w.print("scored {d} predict+update cycles (1st lidar row used only to initialize state)\n", .{n_scored});
    try w.print("singular-matrix updates skipped: {d}\n", .{n_singular});
    try w.print("RMSE      px={d:.4}  py={d:.4}  vx={d:.4}  vy={d:.4}\n", .{
        @sqrt(sum_sq[0] / n), @sqrt(sum_sq[1] / n), @sqrt(sum_sq[2] / n), @sqrt(sum_sq[3] / n),
    });
    try w.print("max |err| px={d:.4}  py={d:.4}  vx={d:.4}  vy={d:.4}\n", .{
        max_abs[0], max_abs[1], max_abs[2], max_abs[3],
    });

    try w.flush();
}
