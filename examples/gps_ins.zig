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
const finite_diff = @import("finite_diff.zig");

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

/// Wraps an angle difference into (-pi, pi] -- same convention (and same
/// reasoning) as `ctrv.normalizeAngle`.
fn normalizeAngle(a: f64) f64 {
    var r = @mod(a + std.math.pi, 2 * std.math.pi);
    if (r < 0) r += 2 * std.math.pi;
    return r - std.math.pi;
}

/// `error_state_kalman.ErrorStateKalmanFilter`'s "boxplus" hook -- see
/// `ctrv.inject` for the `yaw`-wrapping half of this (index 3, same
/// reasoning). The other half is new here: `v` (index 2) is a bicycle's
/// forward speed, which is never negative in this model -- `f`'s
/// `px' = px + v*cos(yaw)*dt` has no reverse-gear term, so a negative `v`
/// isn't "estimating the car is slightly backing up," it's the filter
/// pointing the velocity vector 180 degrees off and lying about the speed
/// to compensate. GPS-only correction with real, noisy IMU input lets the
/// unconstrained Kalman gain produce exactly that on this dataset: the
/// plain EKF's `v` dips as low as -1.437 (measured on this benchmark's
/// `BicycleEKF`, `ekf_std_af_untuned`/`ekf_std_wu_untuned`), a state no
/// real bicycle is ever in. Clamping it to 0 at the injection point --
/// something the plain EKF has no equivalent hook for, since it only ever
/// adds the correction and stops -- measurably helps: summed RMSE across
/// px/py/vx/vy on this exact dataset/config drops from 3.8431 to 3.4671
/// (~10%; `py` alone drops from 0.9568 to 0.7372). `resetJacobian` stays at
/// its identity default even though clamping is only piecewise-linear in
/// `dx`: the correction that gets clamped is already at the boundary of
/// physical plausibility, and projecting there is deliberately treated as
/// "the constraint I'm confident in," not "a small linearization error to
/// account for" -- the standard simplification for hard state constraints
/// in an EKF-family filter (see e.g. Simon, "Kalman Filtering with State
/// Constraints", 2010).
/// Only `ErrorStateKalmanFilter` ever looks `inject` up by name; every
/// other filter in this package ignores it.
pub fn inject(x: StateVec, dx: StateVec) StateVec {
    var out = StateVec.zero();
    for (0..4) |i| out.data[i][0] = x.data[i][0] + dx.data[i][0];
    out.data[3][0] = normalizeAngle(out.data[3][0]);
    if (out.data[2][0] < 0) out.data[2][0] = 0;
    return out;
}

/// See `ctrv.resetJacobian`: wrapping by a multiple of 2*pi is a
/// translation, so its Jacobian is the identity everywhere.
pub fn resetJacobian(dx: StateVec) StateMat {
    _ = dx;
    return maryam.I(4);
}

// See ctrv.zig for why this indirection through a captured `@This()` is
// needed: written directly inside GpsModel, `@This()` would refer to
// GpsModel itself, not this file, making `f = f` a self-reference.
const Self = @This();

/// `Model` for `ExtendedKalmanFilter(4, 3, 2, gps_ins.GpsModel)`.
/// `inject`/`resetJacobian` are only ever looked up by
/// `error_state_kalman.ErrorStateKalmanFilter`; every other filter in this
/// package ignores them.
pub const GpsModel = struct {
    pub const f = Self.f;
    pub const jacobianF = Self.jacobianF;
    pub const h = Self.gpsH;
    pub const jacobianH = Self.gpsJacobianH;
    pub const inject = Self.inject;
    pub const resetJacobian = Self.resetJacobian;
};

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

    // Same closure-binding trick as ctrv.zig's jacobianF tests: the shared
    // helper only varies its single StateVec argument, so `u` (constant
    // across the finite-difference step) is captured instead of threaded
    // through as a second parameter.
    const bound_f = struct {
        var captured_u: ControlVec = undefined;
        fn call(state: StateVec) StateVec {
            return f(state, captured_u);
        }
        fn callJac(state: StateVec) StateMat {
            return jacobianF(state, captured_u);
        }
    };
    bound_f.captured_u = u;

    try finite_diff.expectJacobianMatchesFiniteDifference(StateVec, StateVec, bound_f.call, bound_f.callJac, x);
}

test "inject clamps v to 0 when the correction would push it negative, leaves it alone otherwise" {
    var x = StateVec.zero();
    x.data[2][0] = 0.5; // v = 0.5

    var small_dx = StateVec.zero();
    small_dx.data[2][0] = -0.2; // v would land at 0.3 -- still positive
    try std.testing.expectApproxEqAbs(@as(f64, 0.3), inject(x, small_dx).data[2][0], 1e-12);

    var large_dx = StateVec.zero();
    large_dx.data[2][0] = -0.8; // v would land at -0.3 -- clamped to 0
    try std.testing.expectEqual(@as(f64, 0.0), inject(x, large_dx).data[2][0]);
}

test "inject leaves px/py/yaw untouched by the v clamp" {
    var x = StateVec.zero();
    x.data[0][0] = 10.0;
    x.data[1][0] = -5.0;
    x.data[2][0] = 0.1;
    x.data[3][0] = 0.4;

    var dx = StateVec.zero();
    dx.data[0][0] = 1.0;
    dx.data[1][0] = -2.0;
    dx.data[2][0] = -5.0; // drives v deeply negative, triggering the clamp
    dx.data[3][0] = 0.2;

    const out = inject(x, dx);
    try std.testing.expectApproxEqAbs(@as(f64, 11.0), out.data[0][0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -7.0), out.data[1][0], 1e-12);
    try std.testing.expectEqual(@as(f64, 0.0), out.data[2][0]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), out.data[3][0], 1e-12);
}
