//! Error-State Kalman Filter: same base `Model` interface as
//! `extended_kalman.ExtendedKalmanFilter` (`f`/`jacobianF`/`h`/`jacobianH`,
//! optional `residual`) -- every `Model` already written for the EKF
//! (`ctrv.RadarModel`, `ctrv.LidarModel`, `gps_ins.GpsModel`) works here
//! unchanged, and with those defaults `ErrorStateKalmanFilter` produces
//! *exactly* the same `x`/`P` as `ExtendedKalmanFilter` at every step (see
//! the equivalence test below). What ESKF adds is two more optional `Model`
//! decls that only matter when the state isn't a plain vector the Kalman
//! gain can just be added onto:
//!
//!   - `inject(x: StateVec, dx: StateVec) StateVec` -- the "boxplus"
//!     composition applying a correction `dx` onto the nominal state `x`.
//!     Defaults to plain `x + dx` (identical to the EKF). Override it when a
//!     state component lives on a manifold rather than flat R^n -- the
//!     textbook case is a unit quaternion (`x_new = x (x) exp(dx)`,
//!     renormalized), which this repo has no example of since none of its
//!     models carry one, but the same mechanism also covers the simpler case
//!     already present here: keeping a periodic component (e.g. `yaw`)
//!     wrapped into a fixed range after every correction instead of letting
//!     the raw stored value drift arbitrarily far from it.
//!   - `resetJacobian(dx: StateVec) StateMat` -- the Jacobian of `inject`
//!     with respect to the *error* at the injection point (`G` in the
//!     standard ESKF derivation, e.g. Sola 2017 "Quaternion kinematics for
//!     the error-state Kalman filter", section 6). After injecting a
//!     correction, the covariance carried forward has to account for the
//!     derivative of the composition itself, not just the ordinary Joseph
//!     update: `P' = G @ Joseph(...) @ G^T`. Defaults to the identity, which
//!     is exact whenever `inject` is (or is equivalent to) plain addition --
//!     it's only ever wrong if `inject` is nonlinear in `dx`, so it's opt-in
//!     rather than something every `Model` has to derive.
//!
//! The two together are what "error-state" means: `predict()`'s nonlinear
//! integration and `update()`'s linear gain computation both still work
//! directly in the full state space exactly like the EKF (there is no
//! separately-persisted small-error field) -- `inject`/`resetJacobian` are
//! just the hook `update()` uses to fold the linear correction back into the
//! nominal state through a caller-chosen composition rule instead of always
//! assuming flat vector addition.
//!
//! `jacobianF`/`jacobianH` are reused as-is from the base `Model`
//! (identical to what the EKF already requires): for plain-additive states
//! these already *are* the error-state Jacobians, so there's nothing extra
//! to derive in the cases this repo actually has models for. A `Model` built
//! around a genuinely non-additive `inject` (quaternions) would need
//! `jacobianF`/`jacobianH` computed with respect to the error state rather
//! than the raw state -- a real constraint of this generic implementation,
//! not something papered over.

const std = @import("std");
const maryam = @import("maryam");
const kalman_core = @import("kalman_core.zig");

pub fn ErrorStateKalmanFilter(comptime n: usize, comptime k: usize, comptime m: usize, comptime Model: type) type {
    const Core = kalman_core.KalmanCore(n, m);
    const ControlVec = maryam.MatrixType(k, 1);

    const residual = kalman_core.defaultResidual(Model, Core.MeasureVec);

    const ErrorCorrection = maryam.Equation("K @ y", struct {
        K: Core.GainMat,
        y: Core.MeasureVec,
    });

    const ResetP = maryam.Equation("G @ P @ G^T", struct {
        G: Core.StateMat,
        P: Core.StateMat,
    });

    const inject = if (@hasDecl(Model, "inject")) Model.inject else struct {
        fn call(x: Core.StateVec, dx: Core.StateVec) Core.StateVec {
            return maryam.Equation("x + dx", struct { x: Core.StateVec, dx: Core.StateVec }).eval(.{ .x = x, .dx = dx });
        }
    }.call;

    return struct {
        const Self = @This();

        // Persistent state. Unlike the name might suggest, there's no
        // separate "error state" field here -- the error is only ever a
        // local `dx` inside update(), immediately folded into `x` via
        // `inject` and reset to (implicitly) zero, exactly the point of the
        // reset step.
        x: Core.StateVec, // (n x 1) Nominal state estimate
        P: Core.StateMat, // (n x n) Error-state covariance

        Q: Core.StateMat, // (n x n) Process noise
        R: Core.MeasureNoise, // (m x m) Measurement noise

        pub fn predict(self: *Self, u: ControlVec) void {
            const F = Model.jacobianF(self.x, u);
            self.x = Model.f(self.x, u);
            self.P = Core.PredictP.eval(.{ .F = F, .P = self.P, .Q = self.Q });
        }

        pub fn update(self: *Self, z: Core.MeasureVec) maryam.EvalError!void {
            const H = Model.jacobianH(self.x);
            const S = Core.InnovationS.eval(.{ .H = H, .P = self.P, .R = self.R });
            const K = try Core.KalmanGainK.eval(.{ .S = S, .H = H, .P = self.P });
            const y = residual(z, Model.h(self.x));
            const dx = ErrorCorrection.eval(.{ .K = K, .y = y });

            const P_updated = Core.UpdateP.eval(.{ .I = maryam.I(n), .K = K, .H = H, .P = self.P, .R = self.R });

            self.x = inject(self.x, dx);
            self.P = if (comptime @hasDecl(Model, "resetJacobian"))
                ResetP.eval(.{ .G = Model.resetJacobian(dx), .P = P_updated })
            else
                P_updated;
        }
    };
}

test "matches ExtendedKalmanFilter exactly when inject/resetJacobian use their defaults" {
    const extended_kalman = @import("extended_kalman.zig");
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
    const ESKF = ErrorStateKalmanFilter(1, 1, 1, Model);

    var ekf = EKF{
        .x = Vec1.zero(),
        .P = blk: {
            var mtx = Vec1.zero();
            mtx.data[0][0] = 1;
            break :blk mtx;
        },
        .Q = blk: {
            var mtx = Vec1.zero();
            mtx.data[0][0] = 0.01;
            break :blk mtx;
        },
        .R = blk: {
            var mtx = Vec1.zero();
            mtx.data[0][0] = 0.1;
            break :blk mtx;
        },
    };
    var eskf = ESKF{
        .x = ekf.x,
        .P = ekf.P,
        .Q = ekf.Q,
        .R = ekf.R,
    };

    var u = Vec1.zero();
    u.data[0][0] = 0.2;
    var z = Vec1.zero();

    // 20 predict+update cycles, a mildly moving measurement: enough steps
    // for any accumulated floating-point divergence between the two
    // filters' arithmetic paths to show up if the equivalence claim were
    // only approximate rather than the two filters doing the same
    // computation.
    for (0..20) |i| {
        ekf.predict(u);
        eskf.predict(u);

        z.data[0][0] = 0.4 + 0.01 * @as(f64, @floatFromInt(i));
        try ekf.update(z);
        try eskf.update(z);

        try std.testing.expectApproxEqAbs(ekf.x.data[0][0], eskf.x.data[0][0], 1e-12);
        try std.testing.expectApproxEqAbs(ekf.P.data[0][0], eskf.P.data[0][0], 1e-12);
    }
}

test "custom inject keeps a periodic nominal state bounded while an equivalent EKF's raw state drifts unboundedly, without changing the estimated physical angle" {
    const extended_kalman = @import("extended_kalman.zig");
    const Vec1 = maryam.MatrixType(1, 1);

    // Wraps into (-pi, pi], same convention as ctrv.zig's normalizeAngle.
    const wrap = struct {
        fn of(a: f64) f64 {
            var r = @mod(a + std.math.pi, 2 * std.math.pi);
            if (r < 0) r += 2 * std.math.pi;
            return r - std.math.pi;
        }
    }.of;

    // A 1-state heading tracker: predict() spins the heading forward at a
    // constant rate, update() measures it only through sin(x) (a genuinely
    // periodic quantity), so the *measurement* never needs its own wrap --
    // this isolates the injection mechanism specifically, not
    // residual-wrapping (already covered by ctrv.zig's radarResidual test).
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
        // The only thing distinguishing this Model from a plain EKF one:
        // fold the correction on, then wrap the result back into
        // (-pi, pi] -- resetJacobian is left at its identity default, since
        // wrapping doesn't change the *local* derivative of inject (it's
        // a translation by a multiple of 2*pi, which has Jacobian 1).
        pub fn inject(x: Vec1, dx: Vec1) Vec1 {
            var out = Vec1.zero();
            out.data[0][0] = wrap(x.data[0][0] + dx.data[0][0]);
            return out;
        }
    };

    const EKF = extended_kalman.ExtendedKalmanFilter(1, 1, 1, Model);
    const ESKF = ErrorStateKalmanFilter(1, 1, 1, Model);

    var ekf = EKF{
        .x = Vec1.zero(),
        .P = blk: {
            var mtx = Vec1.zero();
            mtx.data[0][0] = 1;
            break :blk mtx;
        },
        .Q = blk: {
            var mtx = Vec1.zero();
            mtx.data[0][0] = 1e-4;
            break :blk mtx;
        },
        .R = blk: {
            var mtx = Vec1.zero();
            mtx.data[0][0] = 0.05;
            break :blk mtx;
        },
    };
    var eskf = ESKF{
        .x = ekf.x,
        .P = ekf.P,
        .Q = ekf.Q,
        .R = ekf.R,
    };

    var u = Vec1.zero();
    u.data[0][0] = 0.3; // rad/step turn rate -- 500 steps covers ~24 full turns
    const n_steps = 500;

    for (0..n_steps) |i| {
        ekf.predict(u);
        eskf.predict(u);

        const true_heading = 0.3 * @as(f64, @floatFromInt(i + 1));
        var z = Vec1.zero();
        z.data[0][0] = @sin(true_heading);
        try ekf.update(z);
        try eskf.update(z);
    }

    // The point: ESKF's stored state never left the range inject() wraps
    // it into, while the plain EKF's raw additive state grew alongside the
    // number of steps (500 steps * 0.3 rad/step is well past a single
    // revolution, let alone (-pi, pi]).
    try std.testing.expect(@abs(eskf.x.data[0][0]) <= std.math.pi);
    try std.testing.expect(@abs(ekf.x.data[0][0]) > 2 * std.math.pi);

    // And the actual point of calling wrapping "representational": both
    // filters converged on the same physical heading, just represented
    // differently -- wrapping the EKF's raw state lands on (within
    // estimation error of) the ESKF's already-wrapped one.
    try std.testing.expectApproxEqAbs(wrap(ekf.x.data[0][0]), eskf.x.data[0][0], 1e-9);
}

test "resetJacobian actually scales P, not silently ignored" {
    // A deliberately unphysical resetJacobian (G = 2*I) purely to prove
    // update() actually applies it: P after update() should come out to
    // 4x (G@P@G^T with G=2I scales by G^2=4) what the identity-G default
    // would have produced, matching a hand computation, not just "some
    // different number".
    const Vec1 = maryam.MatrixType(1, 1);

    const BaseModel = struct {
        fn f(x: Vec1, u: Vec1) Vec1 {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        fn jacobianF(x: Vec1, u: Vec1) Vec1 {
            _ = x;
            _ = u;
            var m = Vec1.zero();
            m.data[0][0] = 1;
            return m;
        }
        fn h(x: Vec1) Vec1 {
            return x;
        }
        fn jacobianH(x: Vec1) Vec1 {
            _ = x;
            var m = Vec1.zero();
            m.data[0][0] = 1;
            return m;
        }
    };

    const ScaledModel = struct {
        const Self = @This();
        const f = BaseModel.f;
        const jacobianF = BaseModel.jacobianF;
        const h = BaseModel.h;
        const jacobianH = BaseModel.jacobianH;
        fn resetJacobian(dx: Vec1) Vec1 {
            _ = dx;
            var m = Vec1.zero();
            m.data[0][0] = 2;
            return m;
        }
    };

    const Plain = ErrorStateKalmanFilter(1, 1, 1, BaseModel);
    const Scaled = ErrorStateKalmanFilter(1, 1, 1, ScaledModel);

    const p0 = blk: {
        var mtx = Vec1.zero();
        mtx.data[0][0] = 1;
        break :blk mtx;
    };
    const q0 = Vec1.zero();
    const r0 = blk: {
        var mtx = Vec1.zero();
        mtx.data[0][0] = 0.1;
        break :blk mtx;
    };

    var plain = Plain{ .x = Vec1.zero(), .P = p0, .Q = q0, .R = r0 };
    var scaled = Scaled{ .x = Vec1.zero(), .P = p0, .Q = q0, .R = r0 };

    var z = Vec1.zero();
    z.data[0][0] = 0.5;
    try plain.update(z);
    try scaled.update(z);

    try std.testing.expectApproxEqAbs(4.0 * plain.P.data[0][0], scaled.P.data[0][0], 1e-12);
}
