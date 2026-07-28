//! Iterated Extended Kalman Filter: same `predict()` as
//! `extended_kalman.ExtendedKalmanFilter`, but `update()` re-linearizes `H`
//! at successively better state estimates instead of just once at the
//! predicted state -- a Gauss-Newton correction for measurements too
//! nonlinear for a single linearization pass to track well. `Model`'s
//! interface is identical to the EKF's: `f`/`jacobianF`/`h`/`jacobianH`
//! (optional `residual`) -- these are properties of the system being
//! estimated, not of which filter algorithm consumes them, so any `Model`
//! already written for the EKF works here unchanged.
//!
//! Iteration is the standard IEKF/Gauss-Newton form: starting from
//! `x_0 = x_pred` (the ordinary EKF `predict()` result), each step `i`
//! computes
//!   H_i = jacobianH(x_i)
//!   y_i = residual(z, h(x_i)) - H_i @ (x_pred - x_i)
//!   K_i = P_pred @ H_i^T @ (H_i @ P_pred @ H_i^T + R)^-1
//!   x_{i+1} = x_pred + K_i @ y_i
//! for up to `max_iterations` steps, stopping early once `x_{i+1}` stops
//! moving by more than a small fixed tolerance. Only `x` is refined across
//! iterations -- `K_i`/`S_i` always use the *original* predicted `P_pred`,
//! never an iteration-updated one, and `P` itself is only corrected once at
//! the very end, via the same Joseph-form `kalman_core.KalmanCore.UpdateP`
//! every other filter here uses, applied with the *final* iteration's
//! `H`/`K`. `max_iterations = 1` degenerates to exactly the plain EKF
//! update (one linearization, no refinement, `x_pred - x_0 = 0` so the
//! extra correction term vanishes) -- see the benchmark sections in
//! Readme.md for how many iterations these datasets actually benefit from.
//!
//! `max_iterations` is a **comptime** parameter, not a struct field like
//! `Q`/`R`: it's an algorithm-shape choice (how hard to try before giving
//! up), not per-instance data, and being comptime lets several
//! configurations (e.g. 1 vs. 3 vs. 10 iterations) exist side by side as
//! distinct monomorphized types for direct benchmark comparison -- the same
//! reasoning `unscented_kalman.zig` gives for fixing `alpha`/`beta`/`kappa`
//! instead of exposing them.

const std = @import("std");
const maryam = @import("maryam");
const kalman_core = @import("kalman_core.zig");

pub fn IteratedExtendedKalmanFilter(comptime n: usize, comptime k: usize, comptime m: usize, comptime Model: type, comptime max_iterations: usize) type {
    if (max_iterations < 1) @compileError("IteratedExtendedKalmanFilter needs max_iterations >= 1 (1 iteration is exactly the plain EKF update)");

    const Core = kalman_core.KalmanCore(n, m);
    const ControlVec = maryam.MatrixType(k, 1);

    const residual = kalman_core.defaultResidual(Model, Core.MeasureVec);

    // y_i = raw - H_i @ (x_pred - x_i): the correction that accounts for
    // linearizing at x_i instead of x_pred, layered on top of the model's
    // own (possibly angle-wrapped) residual `raw = residual(z, h(x_i))`.
    const IterResidual = maryam.Equation("raw - H @ dx", struct { raw: Core.MeasureVec, H: Core.MeasureMat, dx: Core.StateVec });
    const StateDelta = maryam.Equation("a - b", struct { a: Core.StateVec, b: Core.StateVec });

    // Early-exit tolerance on how much x moves between iterations -- an
    // implementation detail of *when to stop refining*, not a per-instance
    // tunable, same reasoning as fixing `max_iterations` at comptime above.
    const tolerance = 1e-7;

    const maxAbsEntry = struct {
        fn call(v: Core.StateVec) f64 {
            var result: f64 = 0;
            for (0..n) |i| result = @max(result, @abs(v.data[i][0]));
            return result;
        }
    }.call;

    return struct {
        const Self = @This();

        // Persistent State
        x: Core.StateVec, // (n x 1) State estimate
        P: Core.StateMat, // (n x n) Covariance matrix

        Q: Core.StateMat, // (n x n) Process noise
        R: Core.MeasureNoise, // (m x m) Measurement noise

        pub fn predict(self: *Self, u: ControlVec) void {
            const F = Model.jacobianF(self.x, u);
            self.x = Model.f(self.x, u);
            self.P = Core.PredictP.eval(.{ .F = F, .P = self.P, .Q = self.Q });
        }

        pub fn update(self: *Self, z: Core.MeasureVec) maryam.EvalError!void {
            const x_pred = self.x;
            var x_i = x_pred;
            var H_i: Core.MeasureMat = undefined;
            var K_i: Core.GainMat = undefined;

            var iter: usize = 0;
            while (iter < max_iterations) : (iter += 1) {
                H_i = Model.jacobianH(x_i);
                const S = Core.InnovationS.eval(.{ .H = H_i, .P = self.P, .R = self.R });
                K_i = try Core.KalmanGainK.eval(.{ .S = S, .H = H_i, .P = self.P });

                const raw = residual(z, Model.h(x_i));
                const dx = StateDelta.eval(.{ .a = x_pred, .b = x_i });
                const y_i = IterResidual.eval(.{ .raw = raw, .H = H_i, .dx = dx });

                const x_next = Core.ApplyGain.eval(.{ .x = x_pred, .K = K_i, .y = y_i });
                const step = StateDelta.eval(.{ .a = x_next, .b = x_i });
                x_i = x_next;

                if (maxAbsEntry(step) < tolerance) break;
            }

            self.x = x_i;
            self.P = Core.UpdateP.eval(.{ .I = maryam.I(n), .K = K_i, .H = H_i, .P = self.P, .R = self.R });
        }
    };
}

test "max_iterations = 1 exactly reproduces the plain EKF update" {
    // Same 1D h(x) = sin(x) model and same expected numbers as
    // extended_kalman.zig's own test: with only one iteration, x_0 = x_pred
    // always, so the extra "- H @ (x_pred - x_i)" correction term is
    // multiplied by zero and this collapses to exactly the plain EKF.
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var mtx = Vec1.zero();
            mtx.data[0][0] = v;
            return mtx;
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

    const IEKF1 = IteratedExtendedKalmanFilter(1, 1, 1, Model, 1);

    var filter = IEKF1{
        .x = scalar(0),
        .P = scalar(1),
        .Q = scalar(0),
        .R = scalar(0.1),
    };

    filter.predict(scalar(0));
    try std.testing.expectEqual(@as(f64, 0), filter.x.data[0][0]);
    try std.testing.expectEqual(@as(f64, 1), filter.P.data[0][0]);

    // Same closed form as extended_kalman.zig's test: H=1, S=1.1, K=10/11,
    // x = 5/11, P = 1/11.
    try filter.update(scalar(0.5));
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 11.0), filter.x.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 11.0), filter.P.data[0][0], 1e-9);
}

test "max_iterations = 2 re-linearizes at the first iterate and shifts the answer" {
    // Same model as above, but with a second iteration: re-linearizes H at
    // x_1 (the first iteration's result, 5/11) instead of stopping there,
    // and re-derives the expected result independently (not by re-running
    // the implementation's own formula) to confirm the second pass actually
    // changes the answer rather than being a no-op.
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var mtx = Vec1.zero();
            mtx.data[0][0] = v;
            return mtx;
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

    const IEKF2 = IteratedExtendedKalmanFilter(1, 1, 1, Model, 2);

    var filter = IEKF2{
        .x = scalar(0),
        .P = scalar(1),
        .Q = scalar(0),
        .R = scalar(0.1),
    };

    filter.predict(scalar(0));

    // Iteration 1 (x_pred = 0, P_pred = 1): identical to the EKF, x_1 = 5/11.
    const x1 = 5.0 / 11.0;

    // Iteration 2 relinearizes at x_1 (not x_pred): H_2 = cos(x_1). P_pred
    // (not an updated P) is still used for S/K, per the Gauss-Newton form --
    // and S = H^2 * P_pred + R (not P_pred + R): H_2 isn't 1 the way
    // iteration 1's H_1 was, so it doesn't drop out of the square.
    const h2 = @cos(x1);
    const p_pred = 1.0;
    const s2 = h2 * h2 * p_pred + 0.1; // H^2 * P_pred + R
    const k2 = p_pred * h2 / s2;
    const raw2 = 0.5 - @sin(x1); // z - h(x_1)
    const dx2 = 0.0 - x1; // x_pred - x_1
    const y2 = raw2 - h2 * dx2;
    const expected_x2 = 0.0 + k2 * y2; // x_pred + K_2 * y_2
    const expected_p2 = (1.0 - k2 * h2) * (1.0 - k2 * h2) * p_pred + k2 * k2 * 0.1; // Joseph form, scalar

    try filter.update(scalar(0.5));
    try std.testing.expectApproxEqAbs(expected_x2, filter.x.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(expected_p2, filter.P.data[0][0], 1e-9);

    // And the whole point: iterating a second time actually moved x, it
    // wasn't a no-op that just reproduced the first iteration's answer.
    try std.testing.expect(@abs(filter.x.data[0][0] - x1) > 1e-4);
}
