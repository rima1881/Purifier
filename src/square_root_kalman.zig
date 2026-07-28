//! Square-Root Kalman Filter: same `Model` interface as
//! `extended_kalman.ExtendedKalmanFilter` (`f`/`jacobianF`/`h`/`jacobianH`,
//! optional `residual`), but the persistent state is a Cholesky factor `L`
//! of `P` (`P = L @ L^T`) instead of `P` itself. Every `Model` already
//! written for the EKF (`ctrv.RadarModel`, `ctrv.LidarModel`,
//! `gps_ins.GpsModel`) works here unchanged.
//!
//! `update()` is a genuine Potter/Carlson-form square-root update: it never
//! re-forms `P`, propagating `L` directly through a QR decomposition
//! (`operation.qrMatrix`) instead --
//!   R1_2 = a square root of R (R is diagonal-positive in every model this
//!          repo uses, so `operation.choleskyMatrix(R)` always succeeds)
//!   M = [[R1_2^T,     0  ],
//!        [(H @ L)^T,  L^T]]                       -- (m+n) x (m+n)
//!   S  = qrMatrix(M).r                              -- upper-triangular
//!   K  = S[0:m, m:]^T,  N = S[0:m, 0:m]^T,  L' = S[m:, m:]^T
//!   x' = x + K @ N^-1 @ residual(z, h(x))
//! which keeps the working condition number at `cond(L)` instead of
//! `cond(P) = cond(L)^2` -- the actual point of a square-root filter, unlike
//! `predict()` below.
//!
//! `predict()` still round-trips through `P = L @ L^T`, honestly for a
//! specific reason: a real QR-based predict step needs a square root of `Q`
//! (`Q1_2`, so `[F @ L | Q1_2]` can be QR'd the same way `update()`'s `M`
//! is), but every process-noise `Q` this repo actually builds
//! (`ctrv.processNoise`, `gps_ins.processNoise`) is constructed as
//! `G @ Qv @ G^T` from a low-dimensional white-noise source (2 independent
//! noise channels injected into a 4- or 5-dim state) and is therefore
//! always rank-deficient. Neither `operation.choleskyMatrix` (needs strict
//! positive-definiteness) nor `operation.sqrtMatrix` (Denman-Beavers,
//! starts by inverting `Q` itself) can factor a rank-deficient matrix, so
//! there's no `Q1_2` to feed a QR-based predict step for these particular
//! models -- not a maryam gap, a property of the models. `predict()`
//! reconstructs `P` from `L`, runs the *exact same* Joseph-form recursion
//! `kalman_core.KalmanCore` provides, and re-factors the result back into
//! `L` via `choleskyMatrix`, which either succeeds or reports
//! `error.NotPositiveDefinite` -- a real, different benefit (every step's
//! `P` is validated as genuinely SPD, or the filter fails fast) from the
//! numerical-conditioning one `update()` now gets, but not that one.
//!
//! One visible consequence of `update()`'s Householder-based construction:
//! `L`'s diagonal sign is whatever the QR reflection convention happens to
//! produce (not necessarily non-negative the way `predict()`'s
//! Cholesky-derived `L` is). This doesn't affect correctness -- `L` is only
//! ever used via `L @ L^T` or `H @ L`, both sign-invariant -- but a reader
//! inspecting `.L` directly after an `update()` call may see a different
//! sign than after a `predict()` call.

const std = @import("std");
const maryam = @import("maryam");
const kalman_core = @import("kalman_core.zig");
const kalman = @import("kalman.zig");

pub fn SquareRootKalmanFilter(comptime n: usize, comptime k: usize, comptime m: usize, comptime Model: type) type {
    const Core = kalman_core.KalmanCore(n, m);
    const ControlVec = maryam.MatrixType(k, 1);
    const GainN = maryam.MatrixType(m, m);
    const Augmented = maryam.MatrixType(m + n, m + n);

    const residual = kalman_core.defaultResidual(Model, Core.MeasureVec);

    // P = L @ L^T -- the one non-DSL-avoidable reconstruction predict()
    // performs every call (see the module doc comment above for why: Q's
    // rank-deficiency in this repo's models blocks a QR-based predict).
    const ReformP = maryam.Equation("L @ L^T", struct { L: Core.StateMat });

    const HL = maryam.Equation("H @ L", struct { H: Core.MeasureMat, L: Core.StateMat });

    // x + K @ N^-1 @ y, written with the inverse leading into a `@` chain
    // so `Equation` solves `N @ w = y` and computes `K @ w` instead of
    // materializing `N^-1` -- same solve-fusion trick as
    // `KalmanCore.KalmanGainK`.
    const ApplyGain = maryam.Equation("x + K @ N^-1 @ y", struct {
        x: Core.StateVec,
        K: Core.GainMat,
        N: GainN,
        y: Core.MeasureVec,
    });

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

            const R1_2 = maryam.operation.choleskyMatrix(Core.MeasureNoise, self.R) orelse return error.NotPositiveDefinite;
            const hl = HL.eval(.{ .H = H, .L = self.L }); // (m x n)

            var M = Augmented.zero();
            for (0..m) |i| for (0..m) |j| {
                M.data[i][j] = R1_2.data[j][i]; // top-left (m x m): R1_2^T
            };
            for (0..n) |i| for (0..m) |j| {
                M.data[m + i][j] = hl.data[j][i]; // bottom-left (n x m): (H @ L)^T
            };
            for (0..n) |i| for (0..n) |j| {
                M.data[m + i][m + j] = self.L.data[j][i]; // bottom-right (n x n): L^T
            };
            // top-right (m x n) stays zero.

            const qr = maryam.operation.qrMatrix(Augmented, M) orelse return error.RankDeficient;
            const S = qr.r; // upper-triangular (m+n) x (m+n)

            var K: Core.GainMat = undefined; // n x m
            for (0..n) |i| for (0..m) |j| {
                K.data[i][j] = S.data[j][m + i];
            };
            var N: GainN = undefined; // m x m
            for (0..m) |i| for (0..m) |j| {
                N.data[i][j] = S.data[j][i];
            };
            var L_new: Core.StateMat = undefined; // n x n
            for (0..n) |i| for (0..n) |j| {
                L_new.data[i][j] = S.data[m + j][m + i];
            };

            const y = residual(z, Model.h(self.x));
            self.x = try ApplyGain.eval(.{ .x = self.x, .K = K, .N = N, .y = y });
            self.L = L_new;
        }
    };
}

test "1D nonlinear-measurement filter linearizes h(x) = sin(x) at the current estimate, matching the plain EKF" {
    // Same model and same expected numbers as extended_kalman.zig's own
    // test: in exact arithmetic, propagating L (P = L @ L^T) instead of P
    // directly can't change the answer, only how it's computed internally
    // -- so this reuses the EKF test's hand-derived expected values as its
    // own check, rather than re-deriving them. Compares P = L*L (not L
    // itself) after update(): the QR-based update's Householder convention
    // doesn't guarantee L's sign the way predict()'s Cholesky-derived L is
    // (both +L and -L are valid square roots of the same P) -- see the
    // module doc comment.
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
    // y = z - h(x) = 0.5 - 0 = 0.5; x = 0 + K*y = 5/11; P = (1-K)*P = 1/11.
    try filter.update(scalar(0.5));
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 11.0), filter.x.data[0][0], 1e-9);
    const p_new = filter.L.data[0][0] * filter.L.data[0][0];
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 11.0), p_new, 1e-9);
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

test "update() matches the plain EKF on a genuinely multi-dimensional measurement" {
    // The 1D tests above can't exercise the QR block-extraction indexing
    // (K/N/L' slicing only matters once m > 1 or n > 1); this uses a 2-state,
    // 2-measurement linear model (identity F/H) so the expected K/P after
    // one update can be computed by hand via the ordinary (non-square-root)
    // Kalman equations and compared against this filter's QR-based result.
    const Vec2 = maryam.MatrixType(2, 1);
    const Mat2 = maryam.MatrixType(2, 2);

    const vec = struct {
        fn of(a: f64, b: f64) Vec2 {
            var v = Vec2.zero();
            v.data[0][0] = a;
            v.data[1][0] = b;
            return v;
        }
    }.of;
    const diag = struct {
        fn of(a: f64, b: f64) Mat2 {
            var mtx = Mat2.zero();
            mtx.data[0][0] = a;
            mtx.data[1][1] = b;
            return mtx;
        }
    }.of;

    const Model = struct {
        fn f(x: Vec2, u: Vec2) Vec2 {
            _ = u;
            return x;
        }
        fn jacobianF(x: Vec2, u: Vec2) Mat2 {
            _ = x;
            _ = u;
            return diag(1, 1);
        }
        fn h(x: Vec2) Vec2 {
            return x;
        }
        fn jacobianH(x: Vec2) Mat2 {
            _ = x;
            return diag(1, 1);
        }
    };

    const SRKF = SquareRootKalmanFilter(2, 2, 2, Model);

    var filter = SRKF{
        .x = vec(0, 0),
        .L = diag(2, 3), // P0 = diag(4, 9)
        .Q = Mat2.zero(),
        .R = diag(1, 4), // R = diag(1, 4)
    };

    try filter.predict(vec(0, 0)); // F=I, Q=0: x, P unchanged.

    // Ordinary (non-square-root) Kalman math, computed independently: since
    // F=H=I and everything is diagonal, the two state components don't mix
    // and each behaves like the 1D case. P0 = diag(4, 9), R = diag(1, 4):
    // S = P0 + R = diag(5, 13); K = P0/S = diag(4/5, 9/13).
    const k0 = 4.0 / 5.0;
    const k1 = 9.0 / 13.0;
    const z = vec(2.0, -1.0); // y = z - h(x_pred) = z - 0 = z
    const expected_x0 = k0 * 2.0;
    const expected_x1 = k1 * -1.0;
    const expected_p0 = (1 - k0) * 4.0;
    const expected_p1 = (1 - k1) * 9.0;

    try filter.update(z);
    try std.testing.expectApproxEqAbs(expected_x0, filter.x.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(expected_x1, filter.x.data[1][0], 1e-9);

    const P = maryam.operation.mulMatrix(Mat2, Mat2, filter.L, maryam.operation.transposeMatrix(Mat2, filter.L));
    try std.testing.expectApproxEqAbs(expected_p0, P.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(expected_p1, P.data[1][1], 1e-9);
    // Off-diagonal terms should stay ~0: nothing in this model couples the
    // two state components.
    try std.testing.expectApproxEqAbs(@as(f64, 0), P.data[0][1], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0), P.data[1][0], 1e-9);
}

// The tests above only prove SR-KF matches the plain (Joseph-form) filter
// in well-conditioned scenarios -- exactly the case where propagating a
// square root over propagating P directly makes no practical difference.
// The two tests below actually exercise the property the whole technique
// exists for: genuine ill-conditioning, where the two recursions are *not*
// equally accurate in floating point, even though they're the same
// recursion algebraically.
//
// Scenario (both tests): 2-state system, no dynamics (F=I, Q=0), measured
// only through a fixed linear combination H=[1,1] (observing a+b) with tiny
// measurement noise R. This is genuine partial observability, not an
// artificially-constructed pathological matrix: the "sum" direction's
// variance keeps shrinking every update while the "difference" direction
// (never observed) stays at its prior value, driving cond(P) up every
// single step -- the same structural situation that shows up fusing
// redundant/correlated sensors. For a matrix of the form [[a,b],[b,a]]
// (which P stays, by this scenario's symmetry), the eigenvalues are the
// closed form `a+b` (the shrinking, "sum" direction) and `a-b` (the
// untouched "difference" direction) -- no eigenvalue solver needed to
// check either one.

test "ill-conditioned partial-observability: SR-KF stays far closer to the true covariance than Joseph form" {
    // R = 1e-12, 100 predict+update cycles (F=I, Q=0, so predict() is a
    // no-op here -- included anyway to match how these filters are
    // actually driven, rather than isolating update() artificially).
    // Independently verified with a 50-decimal-digit mpmath reference
    // implementing the exact same recursion (see the Python scripts this
    // number was derived from, not included in this repo): after 100
    // steps, P's small ("sum") eigenvalue is 4.999999999999975e-15 --
    // essentially 5e-15 -- and the large ("difference") eigenvalue is
    // exactly 1.0.
    //
    // Measured result: the plain Joseph-form `KalmanFilter`'s small
    // eigenvalue is off by ~45% (stuck around 7.3e-15, no longer tracking
    // the true value once cond(P) gets this large); `SquareRootKalmanFilter`'s
    // is off by ~2.3% (4.9e-15) -- not perfect, but a genuine ~20x
    // reduction in error, from the exact same recursion, computed via QR
    // instead of squaring P directly.
    const StateVec = maryam.MatrixType(2, 1);
    const StateMat = maryam.MatrixType(2, 2);
    const ControlVec = maryam.MatrixType(1, 1);
    const ControlMat = maryam.MatrixType(2, 1);
    const MeasureVec = maryam.MatrixType(1, 1);
    const MeasureMat = maryam.MatrixType(1, 2);
    const MeasureNoise = maryam.MatrixType(1, 1);

    const r = 1e-12;
    const n_steps = 100;
    const ref_small_eig = 4.999999999999975e-15;

    const identity2 = comptime blk: {
        var m = StateMat.zero();
        m.data[0][0] = 1;
        m.data[1][1] = 1;
        break :blk m;
    };
    const sum_h = comptime blk: {
        var m = MeasureMat.zero();
        m.data[0][0] = 1;
        m.data[0][1] = 1;
        break :blk m;
    };
    const noise_r = blk: {
        var m = MeasureNoise.zero();
        m.data[0][0] = r;
        break :blk m;
    };

    // Smaller eigenvalue of a symmetric 2x2 matrix [[a,b],[b,d]]:
    // (a+d)/2 - sqrt(((a-d)/2)^2 + b^2). The simpler shortcut "a+b" only
    // holds when a == d exactly, which isn't guaranteed here -- the
    // QR-based update's Householder reflections can leave the two diagonal
    // entries very slightly different from each other, even though the
    // matrix stays symmetric (b == c).
    const smallEig = struct {
        fn of(p: StateMat) f64 {
            const a = p.data[0][0];
            const d = p.data[1][1];
            const b = p.data[0][1];
            const mid = (a + d) / 2.0;
            const half_diff = (a - d) / 2.0;
            return mid - @sqrt(half_diff * half_diff + b * b);
        }
    }.of;

    // Joseph-form linear KalmanFilter -- the same covariance math every
    // other filter in this repo (except SR-KF's new update()) is built on.
    const KF = kalman.KalmanFilter(2, 1, 1);
    var ekf = KF{
        .x = StateVec.zero(),
        .P = identity2,
        .F = identity2,
        .B = ControlMat.zero(),
        .Q = StateMat.zero(),
        .H = sum_h,
        .R = noise_r,
    };
    for (0..n_steps) |_| {
        ekf.predict(ControlVec.zero());
        try ekf.update(MeasureVec.zero());
    }
    const ekf_small_eig = smallEig(ekf.P);
    const ekf_rel_err = @abs(ekf_small_eig - ref_small_eig) / ref_small_eig;

    // SquareRootKalmanFilter, identical scenario.
    const Model = struct {
        fn f(x: StateVec, u: ControlVec) StateVec {
            _ = u;
            return x;
        }
        fn jacobianF(x: StateVec, u: ControlVec) StateMat {
            _ = x;
            _ = u;
            return identity2;
        }
        fn h(x: StateVec) MeasureVec {
            var out = MeasureVec.zero();
            out.data[0][0] = x.data[0][0] + x.data[1][0];
            return out;
        }
        fn jacobianH(x: StateVec) MeasureMat {
            _ = x;
            return sum_h;
        }
    };

    const SRKF = SquareRootKalmanFilter(2, 1, 1, Model);
    var srkf = SRKF{
        .x = StateVec.zero(),
        .L = identity2, // P0 = I, and I's own Cholesky factor is I
        .Q = StateMat.zero(),
        .R = noise_r,
    };
    for (0..n_steps) |_| {
        try srkf.predict(ControlVec.zero());
        try srkf.update(MeasureVec.zero());
    }
    const srkf_p = maryam.operation.mulMatrix(StateMat, StateMat, srkf.L, maryam.operation.transposeMatrix(StateMat, srkf.L));
    const srkf_small_eig = smallEig(srkf_p);
    const srkf_rel_err = @abs(srkf_small_eig - ref_small_eig) / ref_small_eig;

    // The actual point: once cond(P) gets large enough, the Joseph-form
    // filter gets numerically "stuck" well above the true (still-shrinking)
    // small eigenvalue, while the QR-based square-root update keeps
    // tracking it far more closely. This is not a close call -- expect at
    // least a full order of magnitude difference in relative error (in
    // practice it's closer to 20x -- see the numbers in the comment above).
    try std.testing.expect(srkf_rel_err < ekf_rel_err / 10.0);
    try std.testing.expect(srkf_rel_err < 0.05);
}

test "extreme ill-conditioning: SR-KF reports RankDeficient where Joseph form silently keeps going" {
    // Same scenario, but R = 1e-31 -- small enough that R's own Cholesky
    // factor (~1.8e-16) is at the edge of f64's relative precision (~2.2e-16)
    // compared to the ~1-magnitude entries QR's augmented matrix also
    // contains. maryam's `qrMatrix` has its own tolerance check and returns
    // `null` (surfaced here as `error.RankDeficient`) rather than silently
    // producing a meaningless factor -- confirmed to trigger on the very
    // first `update()` call at this R (see the Python exploration this
    // number came from). The Joseph-form filter has no equivalent check:
    // nothing in `(I - K@H) @ P @ (I - K@H)^T + K@R@K^T` ever asks "is this
    // result still numerically meaningful," so it runs to completion
    // regardless -- not because it's more capable at extreme conditioning,
    // but because it has no mechanism to notice when it isn't.
    const StateVec = maryam.MatrixType(2, 1);
    const StateMat = maryam.MatrixType(2, 2);
    const ControlVec = maryam.MatrixType(1, 1);
    const ControlMat = maryam.MatrixType(2, 1);
    const MeasureVec = maryam.MatrixType(1, 1);
    const MeasureMat = maryam.MatrixType(1, 2);
    const MeasureNoise = maryam.MatrixType(1, 1);

    const r = 1e-31;
    const n_steps = 20;

    const identity2 = comptime blk: {
        var m = StateMat.zero();
        m.data[0][0] = 1;
        m.data[1][1] = 1;
        break :blk m;
    };
    const sum_h = comptime blk: {
        var m = MeasureMat.zero();
        m.data[0][0] = 1;
        m.data[0][1] = 1;
        break :blk m;
    };
    const noise_r = blk: {
        var m = MeasureNoise.zero();
        m.data[0][0] = r;
        break :blk m;
    };

    const Model = struct {
        fn f(x: StateVec, u: ControlVec) StateVec {
            _ = u;
            return x;
        }
        fn jacobianF(x: StateVec, u: ControlVec) StateMat {
            _ = x;
            _ = u;
            return identity2;
        }
        fn h(x: StateVec) MeasureVec {
            var out = MeasureVec.zero();
            out.data[0][0] = x.data[0][0] + x.data[1][0];
            return out;
        }
        fn jacobianH(x: StateVec) MeasureMat {
            _ = x;
            return sum_h;
        }
    };

    const SRKF = SquareRootKalmanFilter(2, 1, 1, Model);
    var srkf = SRKF{
        .x = StateVec.zero(),
        .L = identity2,
        .Q = StateMat.zero(),
        .R = noise_r,
    };
    try srkf.predict(ControlVec.zero());
    try std.testing.expectError(error.RankDeficient, srkf.update(MeasureVec.zero()));

    // Meanwhile the Joseph-form linear KalmanFilter runs the exact same
    // scenario for many more steps without ever erroring.
    const KF = kalman.KalmanFilter(2, 1, 1);
    var ekf = KF{
        .x = StateVec.zero(),
        .P = identity2,
        .F = identity2,
        .B = ControlMat.zero(),
        .Q = StateMat.zero(),
        .H = sum_h,
        .R = noise_r,
    };
    for (0..n_steps) |_| {
        ekf.predict(ControlVec.zero());
        try ekf.update(MeasureVec.zero());
    }
}
