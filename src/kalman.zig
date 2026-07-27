const std = @import("std");
const maryam = @import("maryam");

pub fn KalmanFilter(comptime n: usize, comptime k: usize, comptime m: usize) type {
    // 1. Matrix Type Definitions based on dimensions:
    // n = state count, k = control count, m = measurement count
    const StateVec     = maryam.MatrixType(n, 1);
    const StateMat     = maryam.MatrixType(n, n);
    const ControlVec   = maryam.MatrixType(k, 1);
    const ControlMat   = maryam.MatrixType(n, k);
    const MeasureVec   = maryam.MatrixType(m, 1);
    const MeasureMat   = maryam.MatrixType(m, n);
    const MeasureNoise = maryam.MatrixType(m, m);

    // Derived matrix type for Kalman Gain
    const GainMat      = maryam.MatrixType(n, m);

    return struct {
        const Self = @This();

        // Persistent State
        x: StateVec,      // (n x 1) State estimate
        P: StateMat,      // (n x n) Covariance matrix

        // Model Parameters
        F: StateMat,      // (n x n) State transition
        B: ControlMat,    // (n x k) Control matrix
        Q: StateMat,      // (n x n) Process noise
        H: MeasureMat,    // (m x n) Measurement transition
        R: MeasureNoise,  // (m x m) Measurement noise

        // Symbolic Equations
        const PredictX = maryam.Equation("F @ x + B @ u", struct {
            F: StateMat,
            x: StateVec,
            B: ControlMat,
            u: ControlVec,
        });
        const PredictP = maryam.Equation("F @ P @ F^T + Q", struct {
            F: StateMat,
            P: StateMat,
            Q: StateMat,
        });
        const InnovationS = maryam.Equation("H @ P @ H^T + R", struct {
            H: MeasureMat,
            P: StateMat,
            R: MeasureNoise,
        });
        const KalmanGainK = maryam.Equation("P @ H^T @ S^-1", struct {
            P: StateMat,
            H: MeasureMat,
            S: MeasureNoise,
        });
        const UpdateState = maryam.Equation("x + K @ (z - H @ x)", struct {
            x: StateVec,
            K: GainMat,
            z: MeasureVec,
            H: MeasureMat,
        });
        // Joseph form
        const UpdateP = maryam.Equation("(I - K @ H) @ P @ (I - K @ H)^T + K @ R @ K^T", struct {
            I: StateMat,
            K: GainMat,
            H: MeasureMat,
            P: StateMat,
            R: MeasureNoise,
        });

        pub fn predict(self: *Self, u: ControlVec) void {
            self.x = PredictX.eval(.{ .F = self.F, .x = self.x, .B = self.B, .u = u }) catch unreachable;
            self.P = PredictP.eval(.{ .F = self.F, .P = self.P, .Q = self.Q }) catch unreachable;
        }

        pub fn update(self: *Self, z: MeasureVec) maryam.EvalError!void {
            const S = try InnovationS.eval(.{ .H = self.H, .P = self.P, .R = self.R });
            const K = try KalmanGainK.eval(.{ .P = self.P, .H = self.H, .S = S });
            self.x = UpdateState.eval(.{ .x = self.x, .K = K, .z = z, .H = self.H }) catch unreachable;
            self.P = UpdateP.eval(.{ .I = maryam.I(n), .K = K, .H = self.H, .P = self.P, .R = self.R }) catch unreachable;
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
