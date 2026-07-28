const std = @import("std");
const maryam = @import("maryam");
const kalman_core = @import("kalman_core.zig");

/// Extended Kalman Filter: same covariance recursion as `kalman.KalmanFilter`
/// (see `kalman_core.KalmanCore`) but the state is propagated/corrected
/// through a caller-supplied *nonlinear* model instead of `F @ x` / `H @ x`,
/// and `F`/`H` are the Jacobians of that model evaluated at the current
/// state on every `predict()`/`update()` call instead of fixed matrices.
///
/// `Model` is a comptime namespace (not a runtime value), matching every
/// other generic in this package (`kalman.KalmanFilter`, `maryam.Equation`)
/// being fully monomorphized with no runtime dispatch. `f`/`jacobianF`/`h`/
/// `jacobianH` are properties of the system being estimated (the state
/// transition, its derivative, the measurement function, its derivative) --
/// identical regardless of which filter algorithm consumes them, so every
/// nonlinear filter in this package looks up the same names, not a
/// filter-specific one. `ExtendedKalmanFilter` needs:
///   - `f(x: StateVec, u: ControlVec) StateVec`         -- state transition
///   - `jacobianF(x: StateVec, u: ControlVec) StateMat` -- d(f)/d(x)
///   - `h(x: StateVec) MeasureVec`                       -- measurement model
///   - `jacobianH(x: StateVec) MeasureMat`               -- d(h)/d(x)
/// and may optionally provide:
///   - `residual(z: MeasureVec, h_x: MeasureVec) MeasureVec` -- innovation
///     `z - h(x)`, defaulting to plain subtraction if omitted. Override it
///     when a measurement dimension wraps around (e.g. a bearing angle):
///     naive subtraction near a wraparound boundary can produce a huge
///     spurious residual for what's actually a tiny real difference
///     (measuring -179 degrees against a predicted +179 degrees is off by 2
///     degrees, not 358), which the filter then "corrects" for, diverging.
pub fn ExtendedKalmanFilter(comptime n: usize, comptime k: usize, comptime m: usize, comptime Model: type) type {
    const Core = kalman_core.KalmanCore(n, m);
    const ControlVec = maryam.MatrixType(k, 1);

    const residual = kalman_core.defaultResidual(Model, Core.MeasureVec);

    return struct {
        const Self = @This();

        // Persistent State
        x: Core.StateVec, // (n x 1) State estimate
        P: Core.StateMat, // (n x n) Covariance matrix

        // Noise models. Unlike the linear filter, F/H aren't stored here --
        // they're Model's Jacobians, re-evaluated at the current state on
        // every call, so there's nothing persistent to keep for them.
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
            self.x = Core.ApplyGain.eval(.{ .x = self.x, .K = K, .y = y });
            self.P = Core.UpdateP.eval(.{ .I = maryam.I(n), .K = K, .H = H, .P = self.P, .R = self.R });
        }
    };
}

test "1D nonlinear-measurement filter linearizes h(x) = sin(x) at the current estimate" {
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = v;
            return m;
        }
    }.of;

    const Model = struct {
        // Kept linear on purpose: predict() here only needs to prove the
        // Jacobian-based machinery runs correctly, not exercise nonlinearity
        // twice over. The interesting EKF-specific behavior -- linearizing
        // around the current estimate -- is exercised by h/jacobianH below.
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

    const EKF = ExtendedKalmanFilter(1, 1, 1, Model);

    var filter = EKF{
        .x = scalar(0),
        .P = scalar(1),
        .Q = scalar(0),
        .R = scalar(0.1),
    };

    // predict with u=0: x' = f(0,0) = 0; F = jacobianF(0,0) = 1; P' = P = 1.
    filter.predict(scalar(0));
    try std.testing.expectEqual(@as(f64, 0), filter.x.data[0][0]);
    try std.testing.expectEqual(@as(f64, 1), filter.P.data[0][0]);

    // update against z=0.5: linearized at x=0, where sin(0)=0 and cos(0)=1
    // exactly, so this reduces to the same clean arithmetic as the linear
    // filter's test: H=1, S=P+R=1.1, K=P/S=10/11.
    // y = z - h(x) = 0.5 - 0 = 0.5; x = 0 + K*y = 5/11; P = (1-K)*P = 1/11.
    try filter.update(scalar(0.5));
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 11.0), filter.x.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 11.0), filter.P.data[0][0], 1e-9);
}
