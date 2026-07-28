const maryam = @import("maryam");

/// The covariance-side recursion shared by every filter variant in this
/// package: propagating `P` forward, computing the innovation covariance,
/// the (solve-fused) Kalman gain, applying it to the state, and the
/// Joseph-form covariance update. This is identical whether `F`/`H` are
/// constant/linear (`kalman.KalmanFilter`) or, as in
/// `extended_kalman.ExtendedKalmanFilter`, the Jacobian of a nonlinear model
/// evaluated at the current state each step — only *how* `F`/`H` and the
/// state residual are produced differs between filters, not this part.
pub fn KalmanCore(comptime n: usize, comptime m: usize) type {
    return struct {
        // Exposed so every filter built on top of KalmanCore(n, m) can
        // reuse these instead of redeclaring the same six
        // `maryam.MatrixType(...)` lines -- they used to be copy-pasted
        // into kalman.zig/extended_kalman.zig/iterated_extended_kalman.zig/
        // unscented_kalman.zig/square_root_kalman.zig verbatim.
        pub const StateVec = maryam.MatrixType(n, 1);
        pub const StateMat = maryam.MatrixType(n, n);
        pub const MeasureVec = maryam.MatrixType(m, 1);
        pub const MeasureMat = maryam.MatrixType(m, n);
        pub const MeasureNoise = maryam.MatrixType(m, m);
        pub const GainMat = maryam.MatrixType(n, m);

        pub const PredictP = maryam.Equation("F @ P @ F^T + Q", struct {
            F: StateMat,
            P: StateMat,
            Q: StateMat,
        });

        pub const InnovationS = maryam.Equation("H @ P @ H^T + R", struct {
            H: MeasureMat,
            P: StateMat,
            R: MeasureNoise,
        });

        // K = P @ H^T @ S^-1, written as (S^-1 @ (H @ P))^T instead: since S
        // and P are both symmetric, that's the same result (K^T = S^-1^T @
        // (P @ H^T)^T = S^-1 @ H @ P^T = S^-1 @ H @ P), but it puts the
        // inverse in maryam's `A^-1 @ b` position, so `Equation` solves
        // `S @ y = H @ P` directly instead of materializing `S^-1`.
        pub const KalmanGainK = maryam.Equation("(S^-1 @ (H @ P))^T", struct {
            S: MeasureNoise,
            H: MeasureMat,
            P: StateMat,
        });

        // x_new = x + K @ y, where y is the (already computed) innovation
        // z - H@x (linear KF) or z - h(x) (EKF, nonlinear). Applying the
        // gain to a residual is linear either way, so both filters share it.
        pub const ApplyGain = maryam.Equation("x + K @ y", struct {
            x: StateVec,
            K: GainMat,
            y: MeasureVec,
        });

        // Joseph form: numerically stable even when K isn't exactly the
        // optimal gain (e.g. under floating-point drift) — always yields a
        // symmetric, positive-semidefinite P, unlike the algebraically
        // equivalent but fragile "(I - K @ H) @ P".
        pub const UpdateP = maryam.Equation("(I - K @ H) @ P @ (I - K @ H)^T + K @ R @ K^T", struct {
            I: StateMat,
            K: GainMat,
            H: MeasureMat,
            P: StateMat,
            R: MeasureNoise,
        });
    };
}

/// The `z - h(x)` fallback `ExtendedKalmanFilter` and `UnscentedKalmanFilter`
/// both use when `Model` doesn't supply its own `residual` -- shared here
/// instead of each filter declaring the same `@hasDecl(Model, "residual")`
/// branch and default subtraction. Resolved once at comptime (returns either
/// `Model.residual` itself or a freshly-built default function), so there's
/// no runtime dispatch cost either way.
pub fn defaultResidual(comptime Model: type, comptime MeasureVec: type) fn (MeasureVec, MeasureVec) MeasureVec {
    if (@hasDecl(Model, "residual")) return Model.residual;

    return struct {
        fn call(z: MeasureVec, h_x: MeasureVec) MeasureVec {
            return maryam.Equation("z - h_x", struct { z: MeasureVec, h_x: MeasureVec }).eval(.{ .z = z, .h_x = h_x });
        }
    }.call;
}
