//! Bicycle-model GPS+IMU fusion, driven by real accelerometer/gyro signals
//! (as opposed to `ctrv.zig`, where "control" was a dt-smuggling hack with no
//! real actuation). State [px, py, v, yaw]; control [af, wu, dt] where `af`
//! is forward acceleration and `wu` is yaw rate -- both real IMU channels in
//! the KITTI benchmark this drives (see `kitti_bench.zig`), not derived
//! quantities.
//!
//! f(x, u):
//!   yaw' = yaw + wu*dt
//!   v'   = v + af*dt
//!   px'  = px + v*cos(yaw)*dt
//!   py'  = py + v*sin(yaw)*dt
//!
//! Unlike `ctrv.zig`'s turn-rate state, yaw rate is a measured control input
//! here, not something being estimated -- so there's no yaw_rate-in-the-
//! denominator singularity to special-case, and the Jacobian is a small,
//! unconditional closed form.

const std = @import("std");
const maryam = @import("maryam");

pub const StateVec = maryam.MatrixType(4, 1); // [px, py, v, yaw]
pub const StateMat = maryam.MatrixType(4, 4);
pub const ControlVec = maryam.MatrixType(3, 1); // [af, wu, dt]

pub const GpsVec = maryam.MatrixType(2, 1); // [px, py]
pub const GpsMat = maryam.MatrixType(2, 4);
pub const GpsNoise = maryam.MatrixType(2, 2);

pub fn f(x: StateVec, u: ControlVec) StateVec {
    const af = u.data[0][0];
    const wu = u.data[1][0];
    const dt = u.data[2][0];
    const px = x.data[0][0];
    const py = x.data[1][0];
    const v = x.data[2][0];
    const yaw = x.data[3][0];

    var out = StateVec.zero();
    out.data[0][0] = px + v * @cos(yaw) * dt;
    out.data[1][0] = py + v * @sin(yaw) * dt;
    out.data[2][0] = v + af * dt;
    out.data[3][0] = yaw + wu * dt;
    return out;
}

pub fn jacobianF(x: StateVec, u: ControlVec) StateMat {
    const dt = u.data[2][0];
    const v = x.data[2][0];
    const yaw = x.data[3][0];
    const c = @cos(yaw);
    const s = @sin(yaw);

    var m = StateMat.zero();
    m.data[0][0] = 1;
    m.data[0][2] = c * dt;
    m.data[0][3] = -v * s * dt;
    m.data[1][1] = 1;
    m.data[1][2] = s * dt;
    m.data[1][3] = v * c * dt;
    m.data[2][2] = 1;
    m.data[3][3] = 1;
    return m;
}

/// Q = G @ diag(std_af^2, std_wu^2) @ G^T, where G = d(f)/d([af, wu]) --
/// injecting the IMU's own measurement uncertainty into the state's process
/// noise. Since `af`/`wu` only affect `v`/`yaw` directly within one step
/// (not `px`/`py`, which depend on the *old* `v`/`yaw`), G is diagonal-ish
/// and this reduces to a plain diagonal Q.
pub fn processNoise(dt: f64, std_af: f64, std_wu: f64) StateMat {
    var m = StateMat.zero();
    m.data[2][2] = dt * dt * std_af * std_af;
    m.data[3][3] = dt * dt * std_wu * std_wu;
    return m;
}

pub fn gpsH(x: StateVec) GpsVec {
    var out = GpsVec.zero();
    out.data[0][0] = x.data[0][0];
    out.data[1][0] = x.data[1][0];
    return out;
}

pub fn gpsJacobianH(x: StateVec) GpsMat {
    _ = x;
    var m = GpsMat.zero();
    m.data[0][0] = 1;
    m.data[1][1] = 1;
    return m;
}

// See ctrv.zig for why this indirection through a captured `@This()` is
// needed: written directly inside GpsModel, `@This()` would refer to
// GpsModel itself, not this file, making `f = f` a self-reference.
const Self = @This();

/// `Model` for `ExtendedKalmanFilter(4, 3, 2, gps_ins.GpsModel)`.
pub const GpsModel = struct {
    pub const f = Self.f;
    pub const jacobianF = Self.jacobianF;
    pub const h = Self.gpsH;
    pub const jacobianH = Self.gpsJacobianH;
};

fn expectJacobianMatchesFiniteDifference(x: StateVec, u: ControlVec) !void {
    const eps = 1e-6;
    const analytic = jacobianF(x, u);

    for (0..4) |j| {
        var x_plus = x;
        var x_minus = x;
        x_plus.data[j][0] += eps;
        x_minus.data[j][0] -= eps;

        const f_plus = f(x_plus, u);
        const f_minus = f(x_minus, u);

        for (0..4) |i| {
            const numeric = (f_plus.data[i][0] - f_minus.data[i][0]) / (2 * eps);
            try std.testing.expectApproxEqAbs(numeric, analytic.data[i][j], 1e-4);
        }
    }
}

test "jacobianF matches finite-difference derivatives of f" {
    var x = StateVec.zero();
    x.data[0][0] = 10.0;
    x.data[1][0] = -5.0;
    x.data[2][0] = 4.5;
    x.data[3][0] = -1.2;

    var u = ControlVec.zero();
    u.data[0][0] = 0.8; // af
    u.data[1][0] = 0.15; // wu
    u.data[2][0] = 0.1; // dt

    try expectJacobianMatchesFiniteDifference(x, u);
}
