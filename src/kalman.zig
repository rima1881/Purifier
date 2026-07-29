const std = @import("std");
const maryam = @import("maryam");
const kalman_core = @import("kalman_core.zig");

pub fn KalmanFilter(comptime n: usize, comptime k: usize, comptime m: usize) type {
    const Core = kalman_core.KalmanCore(n, m);
    // Only the k-dependent control types are declared here -- everything
    // else (StateVec/StateMat/MeasureVec/MeasureMat/MeasureNoise) comes
    // straight from Core, since n and m are all it needs.
    const ControlVec = maryam.MatrixType(k, 1);
    const ControlMat = maryam.MatrixType(n, k);

    return struct {
        const Self = @This();

        // Persistent State
        x: Core.StateVec, // (n x 1) State estimate
        P: Core.StateMat, // (n x n) Covariance matrix

        // Model Parameters
        F: Core.StateMat, // (n x n) State transition
        B: ControlMat, // (n x k) Control matrix
        Q: Core.StateMat, // (n x n) Process noise
        H: Core.MeasureMat, // (m x n) Measurement transition
        R: Core.MeasureNoise, // (m x m) Measurement noise

        // Recorded by the most recent update() -- not meaningful to a
        // typical caller directly, but lets a generic wrapper (see
        // `filter_union.FilterKind`/`adaptive_kalman.AdaptiveKalmanFilter`)
        // recover "the gain, residual, and innovation covariance this step
        // actually used" without re-deriving filter-specific math
        // (Jacobian-based here, sigma-point or QR-based for other variants)
        // itself.
        last_K: Core.GainMat = undefined,
        last_y: Core.MeasureVec = undefined,
        last_S: Core.MeasureNoise = undefined,

        // Symbolic Equations specific to the *linear* model (Core covers
        // everything shared with other filter variants).
        const PredictX = maryam.Equation("F @ x + B @ u", struct {
            F: Core.StateMat,
            x: Core.StateVec,
            B: ControlMat,
            u: ControlVec,
        });
        const Innovation = maryam.Equation("z - H @ x", struct {
            z: Core.MeasureVec,
            H: Core.MeasureMat,
            x: Core.StateVec,
        });

        pub fn predict(self: *Self, u: ControlVec) void {
            self.x = PredictX.eval(.{ .F = self.F, .x = self.x, .B = self.B, .u = u });
            self.P = Core.PredictP.eval(.{ .F = self.F, .P = self.P, .Q = self.Q });
        }

        pub fn update(self: *Self, z: Core.MeasureVec) maryam.EvalError!void {
            const S = Core.InnovationS.eval(.{ .H = self.H, .P = self.P, .R = self.R });
            const K = try Core.KalmanGainK.eval(.{ .S = S, .H = self.H, .P = self.P });
            const y = Innovation.eval(.{ .z = z, .H = self.H, .x = self.x });
            self.x = Core.ApplyGain.eval(.{ .x = self.x, .K = K, .y = y });
            self.P = Core.UpdateP.eval(.{ .I = maryam.I(n), .K = K, .H = self.H, .P = self.P, .R = self.R });
            self.last_K = K;
            self.last_y = y;
            self.last_S = S;
        }
    };
}

test "1D constant-velocity-input filter predicts then corrects toward the measurement" {
    const KF = KalmanFilter(1, 1, 1);
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = v;
            return m;
        }
    }.of;

    var filter = KF{
        .x = scalar(0),
        .P = scalar(1),
        .F = scalar(1),
        .B = scalar(1),
        .Q = scalar(0),
        .H = scalar(1),
        .R = scalar(1),
    };

    // predict: x' = F@x + B@u = 0 + 2 = 2; P' = F@P@F^T + Q = 1
    filter.predict(scalar(2));
    try std.testing.expectEqual(@as(f64, 2), filter.x.data[0][0]);
    try std.testing.expectEqual(@as(f64, 1), filter.P.data[0][0]);

    // update against z=3: S = P'+R = 2; K = P'/S = 0.5
    // x = x' + K*(z - x') = 2 + 0.5*1 = 2.5; P = (1-K)*P' = 0.5
    try filter.update(scalar(3));
    try std.testing.expectEqual(@as(f64, 2.5), filter.x.data[0][0]);
    try std.testing.expectEqual(@as(f64, 0.5), filter.P.data[0][0]);
}
