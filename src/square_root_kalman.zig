//! Square-Root Kalman Filter: same `Model` interface and covariance math as
//! `extended_kalman.ExtendedKalmanFilter` (`f`/`jacobianF`/`h`/`jacobianH`,
//! optional `residual`), but the persistent state is a Cholesky factor `L`
//! of `P` (`P = L @ L^T`) instead of `P` itself. Every `Model` already
//! written for the EKF (`ctrv.RadarModel`, `ctrv.LidarModel`,
//! `gps_ins.GpsModel`) works here unchanged.
//!
//! What this buys, honestly: every `predict()`/`update()` call reconstructs
//! `P` from `L`, runs the *exact same* Joseph-form recursion
//! `kalman_core.KalmanCore` already provides (this filter reuses all five of
//! its equations, not just `ApplyGain` like `UnscentedKalmanFilter` does),
//! and then re-factors the result back into `L` via `operation.choleskyMatrix`
//! -- which either succeeds or reports `error.NotPositiveDefinite`. So this
//! is a *fail-fast, checked* filter: any step whose math would produce a
//! non-positive-definite `P` surfaces immediately as an error instead of
//! silently continuing with a `P` that's drifted slightly asymmetric or
//! indefinite under floating-point error, which the plain EKF's Joseph form
//! doesn't detect (Joseph form *tends* to stay PSD, it doesn't guarantee or
//! check it).
//!
//! What this does *not* buy: the classical square-root filter's real
//! numerical-conditioning benefit (the reason the technique exists at all)
//! comes from *never* re-forming `P` -- propagating `L` through a QR
//! decomposition instead, so the effective condition number stays at
//! `cond(L)` rather than `cond(P) = cond(L)^2` the way forming `L @ L^T`
//! here does. That "true" Potter/Carlson/Bierman-style filter needs a QR
//! (or Householder) decomposition primitive, which `maryam` doesn't
//! currently expose (only Cholesky) -- noted in `maryam_fix.md` as the
//! natural next primitive to unblock it. Since this reconstructs `P` fresh
//! from a freshly-validated `L` every step rather than letting error
//! accumulate across many steps before ever checking, it's a real, different
//! benefit (fail-fast validation), just not the one the name classically
//! promises.

const std = @import("std");
const maryam = @import("maryam");
const kalman_core = @import("kalman_core.zig");

pub fn SquareRootKalmanFilter(comptime n: usize, comptime k: usize, comptime m: usize, comptime Model: type) type {
    const Core = kalman_core.KalmanCore(n, m);
    const ControlVec = maryam.MatrixType(k, 1);

    const residual = kalman_core.defaultResidual(Model, Core.MeasureVec);

    // P = L @ L^T -- the one non-DSL-avoidable reconstruction this filter
    // performs every predict()/update() call (see the module doc comment
    // above for why: kalman_core's equations are written in terms of P, not
    // a factor of it).
    const ReformP = maryam.Equation("L @ L^T", struct { L: Core.StateMat });

    return struct {
        const Self = @This();

        // Persistent State
        x: Core.StateVec, // (n x 1) State estimate

        // (n x n) Cholesky factor of the covariance matrix: P = L @ L^T.
        // Callers construct this the same way `operation.choleskyMatrix`
        // would (e.g. `maryam.I(n)` for P = I, since I's own Cholesky factor
        // is I) -- there's no separate "pass P, we'll factor it for you"
        // convenience constructor, matching every other filter in this
        // package being a plain struct literal with no hidden logic.
        L: Core.StateMat,

        Q: Core.StateMat, // (n x n) Process noise
        R: Core.MeasureNoise, // (m x m) Measurement noise

        pub fn predict(self: *Self, u: ControlVec) maryam.EvalError!void {
            const F = Model.jacobianF(self.x, u);
            self.x = Model.f(self.x, u);
            const P = ReformP.eval(.{ .L = self.L });
            const P_pred = Core.PredictP.eval(.{ .F = F, .P = P, .Q = self.Q });
            self.L = maryam.operation.choleskyMatrix(Core.StateMat, P_pred) orelse return error.NotPositiveDefinite;
        }

        pub fn update(self: *Self, z: Core.MeasureVec) maryam.EvalError!void {
            const H = Model.jacobianH(self.x);
            const P = ReformP.eval(.{ .L = self.L });
            const S = Core.InnovationS.eval(.{ .H = H, .P = P, .R = self.R });
            const K = try Core.KalmanGainK.eval(.{ .S = S, .H = H, .P = P });
            const y = residual(z, Model.h(self.x));
            self.x = Core.ApplyGain.eval(.{ .x = self.x, .K = K, .y = y });
            const P_new = Core.UpdateP.eval(.{ .I = maryam.I(n), .K = K, .H = H, .P = P, .R = self.R });
            self.L = maryam.operation.choleskyMatrix(Core.StateMat, P_new) orelse return error.NotPositiveDefinite;
        }
    };
}

test "1D nonlinear-measurement filter linearizes h(x) = sin(x) at the current estimate, matching the plain EKF" {
    // Same model and same expected numbers as extended_kalman.zig's own
    // test: in exact arithmetic, propagating L (P = L @ L^T) instead of P
    // directly can't change the answer, only how it's computed internally
    // -- so this reuses the EKF test's hand-derived expected values as its
    // own check, rather than re-deriving them.
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = v;
            return m;
        }
    }.of;

    const Model = struct {
        fn f(x: Vec1, u: Vec1) Vec1 {
            return scalar(x.data[0][0] + u.data[0][0]);
        }
        fn jacobianF(x: Vec1, u: Vec1) Vec1 {
            _ = x;
            _ = u;
            return scalar(1);
        }
        fn h(x: Vec1) Vec1 {
            return scalar(@sin(x.data[0][0]));
        }
        fn jacobianH(x: Vec1) Vec1 {
            return scalar(@cos(x.data[0][0]));
        }
    };

    const SRKF = SquareRootKalmanFilter(1, 1, 1, Model);

    var filter = SRKF{
        .x = scalar(0),
        .L = scalar(1), // P = L @ L^T = 1, same as the EKF test's P = 1
        .Q = scalar(0),
        .R = scalar(0.1),
    };

    // predict with u=0: x' = f(0,0) = 0; F = jacobianF(0,0) = 1;
    // P' = F@P@F^T + Q = 1, so L' = cholesky(1) = 1.
    try filter.predict(scalar(0));
    try std.testing.expectEqual(@as(f64, 0), filter.x.data[0][0]);
    try std.testing.expectApproxEqAbs(@as(f64, 1), filter.L.data[0][0], 1e-9);

    // update against z=0.5: linearized at x=0, where sin(0)=0 and cos(0)=1
    // exactly, so H=1, S=P+R=1.1, K=P/S=10/11.
    // y = z - h(x) = 0.5 - 0 = 0.5; x = 0 + K*y = 5/11; P = (1-K)*P = 1/11,
    // so L = sqrt(1/11).
    try filter.update(scalar(0.5));
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 11.0), filter.x.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(1.0 / 11.0)), filter.L.data[0][0], 1e-9);
}

test "NotPositiveDefinite surfaces when the covariance recursion would produce an indefinite P" {
    // Directly exercises the failure path this filter adds over the plain
    // EKF: force an indefinite P (Q with a negative diagonal entry -- no
    // real covariance matrix looks like that, but nothing stops a caller
    // from constructing one) and check update()/predict() reports it
    // instead of silently accepting a broken factor.
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = v;
            return m;
        }
    }.of;

    const Model = struct {
        fn f(x: Vec1, u: Vec1) Vec1 {
            _ = u;
            return x;
        }
        fn jacobianF(x: Vec1, u: Vec1) Vec1 {
            _ = x;
            _ = u;
            return scalar(1);
        }
        fn h(x: Vec1) Vec1 {
            return x;
        }
        fn jacobianH(x: Vec1) Vec1 {
            _ = x;
            return scalar(1);
        }
    };

    const SRKF = SquareRootKalmanFilter(1, 1, 1, Model);

    var filter = SRKF{
        .x = scalar(0),
        .L = scalar(0),
        .Q = blk: {
            var neg = Vec1.zero();
            neg.data[0][0] = -1;
            break :blk neg;
        },
        .R = scalar(0.1),
    };

    // P' = F@P@F^T + Q = 0 + (-1) = -1: not positive-definite, so the
    // Cholesky re-factorization inside predict() must fail rather than
    // silently storing a nonsensical L.
    try std.testing.expectError(error.NotPositiveDefinite, filter.predict(scalar(0)));
}
