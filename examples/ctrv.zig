//! Constant Turn Rate and Velocity (CTRV) motion model for the Extended
//! Kalman Filter benchmark: state [px, py, v, yaw, yaw_rate]. Unlike the
//! constant-velocity model `imu_bench.zig` uses for the linear filter, this
//! actually turns -- and its radar measurement model is genuinely nonlinear,
//! so an EKF built on it can consume the dataset's radar rows, which the
//! linear filter has to skip entirely.
//!
//! Jacobians are the standard published ones for this exact model (see e.g.
//! Udacity's Sensor Fusion course notes); `jacobianF`/`radarJacobianH` are
//! checked against finite-difference numerical derivatives in the tests
//! below, since a wrong sign here fails silently (the filter just diverges)
//! rather than with a compile or runtime error.

const std = @import("std");
const maryam = @import("maryam");

pub const StateVec = maryam.MatrixType(5, 1); // [px, py, v, yaw, yaw_rate]
pub const StateMat = maryam.MatrixType(5, 5);

// The generic `ExtendedKalmanFilter`'s predict() only takes a "control"
// vector, with no dedicated timestep parameter -- so, same trick as
// `imu_bench.zig` already uses for the linear filter's dt-dependent F/Q,
// dt is smuggled through as a 1x1 "control" input here. There's no actual
// external actuation in this model.
pub const DtVec = maryam.MatrixType(1, 1);

pub const LidarVec = maryam.MatrixType(2, 1); // [px, py]
pub const LidarMat = maryam.MatrixType(2, 5);
pub const LidarNoise = maryam.MatrixType(2, 2);

pub const RadarVec = maryam.MatrixType(3, 1); // [rho, theta, rho_dot]
pub const RadarMat = maryam.MatrixType(3, 5);
pub const RadarNoise = maryam.MatrixType(3, 3);

const yaw_rate_eps = 1e-4;
const rho_eps = 1e-6;

pub fn f(x: StateVec, dt_vec: DtVec) StateVec {
    const dt = dt_vec.data[0][0];
    const px = x.data[0][0];
    const py = x.data[1][0];
    const v = x.data[2][0];
    const yaw = x.data[3][0];
    const yawd = x.data[4][0];

    var out = StateVec.zero();
    if (@abs(yawd) > yaw_rate_eps) {
        out.data[0][0] = px + (v / yawd) * (@sin(yaw + yawd * dt) - @sin(yaw));
        out.data[1][0] = py + (v / yawd) * (-@cos(yaw + yawd * dt) + @cos(yaw));
    } else {
        out.data[0][0] = px + v * @cos(yaw) * dt;
        out.data[1][0] = py + v * @sin(yaw) * dt;
    }
    out.data[2][0] = v;
    out.data[3][0] = yaw + yawd * dt;
    out.data[4][0] = yawd;
    return out;
}

pub fn jacobianF(x: StateVec, dt_vec: DtVec) StateMat {
    const dt = dt_vec.data[0][0];
    const v = x.data[2][0];
    const yaw = x.data[3][0];
    const yawd = x.data[4][0];

    var m = StateMat.zero();
    m.data[0][0] = 1;
    m.data[1][1] = 1;
    m.data[2][2] = 1;
    m.data[3][3] = 1;
    m.data[3][4] = dt;
    m.data[4][4] = 1;

    const s0 = @sin(yaw);
    const c0 = @cos(yaw);

    if (@abs(yawd) > yaw_rate_eps) {
        const s1 = @sin(yaw + yawd * dt);
        const c1 = @cos(yaw + yawd * dt);

        m.data[0][2] = (s1 - s0) / yawd;
        m.data[0][3] = (v / yawd) * (c1 - c0);
        m.data[0][4] = v * dt * c1 / yawd - v * (s1 - s0) / (yawd * yawd);

        m.data[1][2] = (-c1 + c0) / yawd;
        m.data[1][3] = (v / yawd) * (s1 - s0);
        m.data[1][4] = v * dt * s1 / yawd - v * (-c1 + c0) / (yawd * yawd);
    } else {
        // Straight-line limit. d(px')/d(yaw_rate) and d(py')/d(yaw_rate) are
        // second-order in yaw_rate here (~yaw_rate * dt^2), negligible right
        // where this branch is taken (yaw_rate ~ 0, dt small) -- dropped
        // rather than carrying the extra term through the limit.
        m.data[0][2] = c0 * dt;
        m.data[0][3] = -v * s0 * dt;
        m.data[1][2] = s0 * dt;
        m.data[1][3] = v * c0 * dt;
    }

    return m;
}

/// Q = G @ Qv @ G^T, mapping independent longitudinal/yaw-acceleration white
/// noise (variances `std_a^2`, `std_yawdd^2`) into the 5-dim state. Built
/// directly from its entries (not via `maryam.Equation`) since G is
/// constructed from scratch out of trig terms, not combined from existing
/// matrices.
pub fn processNoise(x: StateVec, dt: f64, std_a: f64, std_yawdd: f64) StateMat {
    const yaw = x.data[3][0];
    const dt2 = dt * dt;

    const g02 = 0.5 * dt2 * @cos(yaw);
    const g12 = 0.5 * dt2 * @sin(yaw);
    const g22 = dt;
    const g31 = 0.5 * dt2;
    const g41 = dt;

    const va2 = std_a * std_a;
    const vd2 = std_yawdd * std_yawdd;

    var m = StateMat.zero();
    m.data[0][0] = g02 * g02 * va2;
    m.data[0][1] = g02 * g12 * va2;
    m.data[0][2] = g02 * g22 * va2;
    m.data[1][0] = g12 * g02 * va2;
    m.data[1][1] = g12 * g12 * va2;
    m.data[1][2] = g12 * g22 * va2;
    m.data[2][0] = g22 * g02 * va2;
    m.data[2][1] = g22 * g12 * va2;
    m.data[2][2] = g22 * g22 * va2;
    m.data[3][3] = g31 * g31 * vd2;
    m.data[3][4] = g31 * g41 * vd2;
    m.data[4][3] = g41 * g31 * vd2;
    m.data[4][4] = g41 * g41 * vd2;
    return m;
}

pub fn lidarH(x: StateVec) LidarVec {
    var out = LidarVec.zero();
    out.data[0][0] = x.data[0][0];
    out.data[1][0] = x.data[1][0];
    return out;
}

pub fn lidarJacobianH(x: StateVec) LidarMat {
    _ = x;
    var m = LidarMat.zero();
    m.data[0][0] = 1;
    m.data[1][1] = 1;
    return m;
}

/// Wraps an angle difference into (-pi, pi]: measuring a bearing of +179
/// degrees against a predicted -179 degrees is a 2-degree difference, not
/// 358 -- `atan2` return values straddle a +-pi branch cut that plain
/// subtraction doesn't know about.
fn normalizeAngle(a: f64) f64 {
    var r = @mod(a + std.math.pi, 2 * std.math.pi);
    if (r < 0) r += 2 * std.math.pi;
    return r - std.math.pi;
}

/// Radar's `z - h(x)` residual, with the bearing (theta) component
/// angle-wrapped -- see `extended_kalman.ExtendedKalmanFilter.residual`.
pub fn radarResidual(z: RadarVec, h_x: RadarVec) RadarVec {
    var y = maryam.operation.subMatrix(RadarVec, z, h_x);
    y.data[1][0] = normalizeAngle(y.data[1][0]);
    return y;
}

pub fn radarH(x: StateVec) RadarVec {
    const px = x.data[0][0];
    const py = x.data[1][0];
    const v = x.data[2][0];
    const yaw = x.data[3][0];
    const vx = v * @cos(yaw);
    const vy = v * @sin(yaw);

    const rho = @sqrt(px * px + py * py);
    var out = RadarVec.zero();
    out.data[0][0] = rho;
    out.data[1][0] = std.math.atan2(py, px);
    out.data[2][0] = if (rho > rho_eps) (px * vx + py * vy) / rho else 0;
    return out;
}

pub fn radarJacobianH(x: StateVec) RadarMat {
    const px = x.data[0][0];
    const py = x.data[1][0];
    const v = x.data[2][0];
    const yaw = x.data[3][0];
    const vx = v * @cos(yaw);
    const vy = v * @sin(yaw);

    const rho2 = px * px + py * py;
    const rho = @sqrt(rho2);

    var m = RadarMat.zero();
    // Degenerate at the origin (never occurs in this dataset); leave H = 0
    // rather than dividing by ~0.
    if (rho < rho_eps) return m;
    const rho3 = rho2 * rho;

    m.data[0][0] = px / rho;
    m.data[0][1] = py / rho;

    m.data[1][0] = -py / rho2;
    m.data[1][1] = px / rho2;

    m.data[2][0] = py * (vx * py - vy * px) / rho3;
    m.data[2][1] = px * (vy * px - vx * py) / rho3;
    m.data[2][2] = (px * @cos(yaw) + py * @sin(yaw)) / rho;
    m.data[2][3] = v * (py * @cos(yaw) - px * @sin(yaw)) / rho;
    // d(rho_dot)/d(yaw_rate) = 0 (yaw_rate doesn't affect the model's
    // instantaneous velocity direction), so column 4 stays zero.

    return m;
}

// `@This()` captured here (outside of LidarModel/RadarModel) refers to this
// file's own top-level namespace, so `Self.f` below unambiguously means
// ctrv.zig's `f` -- inside LidarModel/RadarModel themselves, `@This()` would
// refer to that inner struct instead, which is what `extended_kalman.ExtendedKalmanFilter`'s
// generic `Model` parameter is looking for by name (`f`, `jacobianF`, ...),
// so a same-named alias can't just point at itself.
const Self = @This();

/// `Model` for `ExtendedKalmanFilter(5, 1, 2, ctrv.LidarModel)`: lidar's
/// measurement is linear (direct px/py), so no residual override is needed.
pub const LidarModel = struct {
    pub const f = Self.f;
    pub const jacobianF = Self.jacobianF;
    pub const h = Self.lidarH;
    pub const jacobianH = Self.lidarJacobianH;
};

/// `Model` for `ExtendedKalmanFilter(5, 1, 3, ctrv.RadarModel)`: radar's
/// bearing measurement wraps around, so this supplies `residual` too.
pub const RadarModel = struct {
    pub const f = Self.f;
    pub const jacobianF = Self.jacobianF;
    pub const h = Self.radarH;
    pub const jacobianH = Self.radarJacobianH;
    pub const residual = Self.radarResidual;
};

fn expectJacobianMatchesFiniteDifference(
    comptime OutVec: type,
    func: *const fn (StateVec) OutVec,
    jac: *const fn (StateVec) maryam.MatrixType(OutVec.nrows, 5),
    x: StateVec,
) !void {
    const eps = 1e-6;
    const analytic = jac(x);

    for (0..5) |j| {
        var x_plus = x;
        var x_minus = x;
        x_plus.data[j][0] += eps;
        x_minus.data[j][0] -= eps;

        const f_plus = func(x_plus);
        const f_minus = func(x_minus);

        for (0..OutVec.nrows) |i| {
            const numeric = (f_plus.data[i][0] - f_minus.data[i][0]) / (2 * eps);
            try std.testing.expectApproxEqAbs(numeric, analytic.data[i][j], 1e-4);
        }
    }
}

test "radarJacobianH matches finite-difference derivatives of radarH" {
    var x = StateVec.zero();
    x.data[0][0] = 120.0; // px
    x.data[1][0] = -45.0; // py
    x.data[2][0] = 12.0; // v
    x.data[3][0] = 0.7; // yaw
    x.data[4][0] = 0.05; // yaw_rate (unused by radarH, kept nonzero to be realistic)

    try expectJacobianMatchesFiniteDifference(RadarVec, radarH, radarJacobianH, x);
}

test "jacobianF matches finite-difference derivatives of f, turning case" {
    const dt: DtVec = .{ .data = .{.{0.1}} };
    var x = StateVec.zero();
    x.data[0][0] = 10.0;
    x.data[1][0] = 5.0;
    x.data[2][0] = 8.0; // v
    x.data[3][0] = -0.4; // yaw
    x.data[4][0] = 0.3; // yaw_rate (well above the straight-line threshold)

    const bound_f = struct {
        var captured_dt: DtVec = undefined;
        fn call(state: StateVec) StateVec {
            return f(state, captured_dt);
        }
        fn callJac(state: StateVec) StateMat {
            return jacobianF(state, captured_dt);
        }
    };
    bound_f.captured_dt = dt;

    try expectJacobianMatchesFiniteDifference(StateVec, bound_f.call, bound_f.callJac, x);
}

test "jacobianF matches finite-difference derivatives of f, near-straight-line case" {
    const dt: DtVec = .{ .data = .{.{0.1}} };
    var x = StateVec.zero();
    x.data[0][0] = 10.0;
    x.data[1][0] = 5.0;
    x.data[2][0] = 8.0;
    x.data[3][0] = 1.1;
    x.data[4][0] = 0.0; // straight-line branch

    const bound_f = struct {
        var captured_dt: DtVec = undefined;
        fn call(state: StateVec) StateVec {
            return f(state, captured_dt);
        }
        fn callJac(state: StateVec) StateMat {
            return jacobianF(state, captured_dt);
        }
    };
    bound_f.captured_dt = dt;

    try expectJacobianMatchesFiniteDifference(StateVec, bound_f.call, bound_f.callJac, x);
}

test "radarResidual wraps the bearing across the +-pi branch cut" {
    // Measured ~+179 degrees vs. predicted ~-179 degrees: physically a ~2
    // degree difference, not the ~358 degrees plain subtraction would give.
    // This is the exact failure mode that showed up against the real
    // benchmark dataset: naive z.theta - h(x).theta produced a residual of
    // ~2*pi, which the filter then "corrected" for, diverging badly.
    var z = RadarVec.zero();
    z.data[0][0] = 6.0;
    z.data[1][0] = 3.190031;
    z.data[2][0] = 1.776367;

    var h_x = RadarVec.zero();
    h_x.data[0][0] = 5.9;
    h_x.data[1][0] = -3.136178;
    h_x.data[2][0] = 1.8;

    const y = radarResidual(z, h_x);

    try std.testing.expect(@abs(y.data[1][0]) < 0.1);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0430237), y.data[1][0], 1e-6);
    // rho/rho_dot residuals are untouched by the wrap.
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), y.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, -0.023633), y.data[2][0], 1e-6);
}

test "normalizeAngle handles multi-wrap and exact-boundary inputs" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), normalizeAngle(4 * std.math.pi), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), normalizeAngle(0.1 - 4 * std.math.pi), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, -std.math.pi + 0.1), normalizeAngle(std.math.pi + 0.1), 1e-9);
}
