//! Unscented Kalman Filter: propagates a small, deterministically-chosen set
//! of "sigma points" through the actual nonlinear `f`/`h` (no linearization,
//! no derivatives) and reconstructs mean/covariance from their weighted
//! statistics. `Model` must provide `f(x, u) -> StateVec` and
//! `h(x) -> MeasureVec` -- no `jacobianF`/`jacobianH`, unlike
//! `extended_kalman.ExtendedKalmanFilter`. This means any `Model` already
//! written for the EKF works here unchanged (the extra `jacobianF`/
//! `jacobianH` decls are simply never referenced).
//!
//! Uses the standard additive-noise formulation: sigma points come from the
//! Cholesky factor of `P` alone (not augmented with `Q`/`R`), and process/
//! measurement noise are added directly to the propagated covariances --
//! same convention `KalmanFilter`/`ExtendedKalmanFilter` already use for
//! `Q`/`R`, so switching between filters doesn't change how noise is
//! modeled.
//!
//! Sigma-point spread parameters (`alpha=1, beta=2, kappa=0`, the standard
//! Van der Merwe / Wan parameterization, chosen for `lambda=0` -- avoids the
//! huge weight magnitudes a tiny `alpha` produces) are fixed internal
//! constants, not part of the public API: they define sigma-point geometry,
//! not a per-instance tunable like `Q`/`R`.
//!
//! The final covariance update, `P = P - K @ S @ K^T`, is the textbook
//! unscented form (matches every reference implementation, including
//! filterpy's) -- unlike `KalmanCore`'s Joseph-form update, it isn't
//! guaranteed to stay positive-semidefinite under floating-point error, and
//! `update()` symmetrizes the result (`P = 0.5 * (P + P^T)`) afterward to at
//! least guarantee the symmetry half, since `predict()`'s sigma points need
//! a Cholesky factor of `P` and `choleskyMatrix` requires an (exactly)
//! symmetric input. A square-root UKF (propagating a Cholesky factor of `P`
//! directly, sidestepping the PSD half of this too) is on the roadmap in
//! Readme.md but not implemented here.
//!
//! Every fixed-shape matrix computation below goes through `maryam.Equation`
//! rather than calling `maryam.operation.*` directly, matching
//! `kalman_core.KalmanCore`'s style -- see the `Eq` namespace. The one
//! exception is `operation.choleskyMatrix`: the equation DSL's `^0.5` is a
//! *symmetric* matrix square root (`sqrtMatrix`), a different decomposition
//! from the lower-triangular Cholesky factor sigma points actually need, and
//! the DSL has no syntax for the latter.

const std = @import("std");
const maryam = @import("maryam");
const kalman_core = @import("kalman_core.zig");

pub fn UnscentedKalmanFilter(comptime n: usize, comptime k: usize, comptime m: usize, comptime Model: type) type {
    const Core = kalman_core.KalmanCore(n, m);
    const ControlVec = maryam.MatrixType(k, 1);
    // Same shape as Core.GainMat (n x m) -- kept as its own name since
    // "cross-covariance" is what it actually is in the UKF's own math (see
    // Eq.CrossAccumulate below), not a Kalman gain yet at that point.
    const CrossCov = Core.GainMat;

    const num_sigma = 2 * n + 1;
    const nf: f64 = @floatFromInt(n);

    const alpha = 1.0;
    const beta = 2.0;
    const kappa = 0.0;
    const lambda = alpha * alpha * (nf + kappa) - nf;
    const spread: f64 = @sqrt(nf + lambda);

    // `wm`/`wc` need a loop to build, but a `var` can't be captured into the
    // methods of the `struct { ... }` returned below (only `const` values
    // cross that namespace boundary) -- so build them in a block and only
    // bind the finished, immutable array value.
    const wm: [num_sigma]f64 = blk: {
        var w: [num_sigma]f64 = undefined;
        w[0] = lambda / (nf + lambda);
        for (1..num_sigma) |i| w[i] = 1.0 / (2.0 * (nf + lambda));
        break :blk w;
    };
    const wc: [num_sigma]f64 = blk: {
        var w: [num_sigma]f64 = undefined;
        w[0] = wm[0] + (1.0 - alpha * alpha + beta);
        for (1..num_sigma) |i| w[i] = wm[i];
        break :blk w;
    };

    // Named DSL equations for every fixed-shape matrix computation this
    // filter performs, mirroring `kalman_core.KalmanCore`: each name
    // documents *what* the recursion computes, while `Equation` handles the
    // add/sub/mul/transpose/solve mechanics (including fusing a trailing
    // `S^-1 @ ...` into a solve instead of materializing an inverse).
    const Eq = struct {
        // Sigma points: mean +- spread * (Cholesky column). Column
        // extraction itself needs indexing the DSL can't express, so that
        // stays a manual loop in `sigmaPoints` below; only the final
        // add/subtract goes through here.
        const SigmaPlus = maryam.Equation("mean + delta", struct { mean: Core.StateVec, delta: Core.StateVec });
        const SigmaMinus = maryam.Equation("mean - delta", struct { mean: Core.StateVec, delta: Core.StateVec });

        // A sigma point's deviation from a reconstructed mean -- reused for
        // the state (predict) and, in update(), both the prior-state
        // deviation and the measurement deviation (`V` covers both shapes).
        fn Deviation(comptime V: type) type {
            return maryam.Equation("point - mean", struct { point: V, mean: V });
        }

        // Weighted-sum reconstruction of a mean from sigma points, reused
        // for both the state mean (predict) and the measurement mean
        // (update) via `V`.
        fn WeightedAccumulate(comptime V: type) type {
            return maryam.Equation("acc + w * point", struct { acc: V, w: f64, point: V });
        }

        // Weighted-sum reconstruction of a covariance from sigma-point
        // deviations, `acc + w * (diff @ diff^T)` -- reused for the state
        // covariance and the innovation covariance `S` via `Acc`/`V`.
        fn WeightedOuterAccumulate(comptime Acc: type, comptime V: type) type {
            return maryam.Equation("acc + w * (diff @ diff^T)", struct { acc: Acc, w: f64, diff: V });
        }

        // Same idea for the state/measurement cross-covariance `Pxz`, whose
        // two operands aren't the same shape, so it can't reuse
        // `WeightedOuterAccumulate`.
        const CrossAccumulate = maryam.Equation("acc + w * (x_diff @ z_diff^T)", struct {
            acc: CrossCov,
            w: f64,
            x_diff: Core.StateVec,
            z_diff: Core.MeasureVec,
        });

        // K = Pxz @ S^-1, written as (S^-1 @ Pxz^T)^T instead: since S is
        // symmetric, that's the same result, but it puts the inverse in
        // `Equation`'s fused-solve position, so this solves `S @ y = Pxz^T`
        // for `y` directly instead of materializing `S^-1` -- same trick as
        // `KalmanCore.KalmanGainK`.
        const GainK = maryam.Equation("(S^-1 @ Pxz^T)^T", struct { S: Core.MeasureNoise, Pxz: CrossCov });

        // Textbook (non-Joseph) covariance update -- see the module doc
        // comment above for why this, unlike `KalmanCore.UpdateP`, isn't
        // guaranteed to stay positive-semidefinite under floating-point
        // error. `Symmetrize` (applied right after, in `update()` below)
        // fixes the symmetry half of that: `K @ S @ K^T` is symmetric in
        // exact arithmetic, but the matrix-multiply order `Equation`
        // generates for it isn't guaranteed to produce `P[i][j] ==
        // P[j][i]` bit-for-bit once rounding is involved, and a `P` that's
        // even slightly asymmetric will fail `choleskyMatrix` the next time
        // `predict()` tries to draw sigma points from it. This doesn't
        // restore the PSD guarantee Joseph form has -- a genuinely negative
        // eigenvalue introduced by rounding would still be a negative
        // eigenvalue after averaging with its own transpose -- only the
        // symmetry half.
        const UpdateP = maryam.Equation("P - K @ S @ K^T", struct { P: Core.StateMat, K: CrossCov, S: Core.MeasureNoise });
        const Symmetrize = maryam.Equation("0.5 * (P + P^T)", struct { P: Core.StateMat });

        const AddProcessNoise = maryam.Equation("cov + Q", struct { cov: Core.StateMat, Q: Core.StateMat });
        const AddMeasureNoise = maryam.Equation("s + R", struct { s: Core.MeasureNoise, R: Core.MeasureNoise });
    };

    const residual = kalman_core.defaultResidual(Model, Core.MeasureVec);

    return struct {
        const Self = @This();

        // Persistent State
        x: Core.StateVec, // (n x 1) State estimate
        P: Core.StateMat, // (n x n) Covariance matrix

        Q: Core.StateMat, // (n x n) Process noise
        R: Core.MeasureNoise, // (m x m) Measurement noise

        // Recorded by the most recent update() -- see kalman.zig's
        // `last_K`/`last_y`/`last_S` fields for why this exists on every
        // filter variant (generic wrappers like `adaptive_kalman.AdaptiveKalmanFilter`).
        last_K: Core.GainMat = undefined,
        last_y: Core.MeasureVec = undefined,
        last_S: Core.MeasureNoise = undefined,

        // Sigma points from the *predicted* state, stashed by predict() and
        // consumed by update() -- the standard UKF reuses these rather than
        // regenerating fresh ones from the already-Q-inflated post-predict P.
        sigma_points: [num_sigma]Core.StateVec = undefined,

        fn sigmaPoints(mean: Core.StateVec, cov: Core.StateMat) maryam.EvalError![num_sigma]Core.StateVec {
            const l = maryam.operation.choleskyMatrix(Core.StateMat, cov) orelse return error.NotPositiveDefinite;

            var pts: [num_sigma]Core.StateVec = undefined;
            pts[0] = mean;
            for (0..n) |col| {
                var delta = Core.StateVec.zero();
                for (0..n) |row| delta.data[row][0] = l.data[row][col] * spread;
                pts[1 + col] = Eq.SigmaPlus.eval(.{ .mean = mean, .delta = delta });
                pts[1 + n + col] = Eq.SigmaMinus.eval(.{ .mean = mean, .delta = delta });
            }
            return pts;
        }

        pub fn predict(self: *Self, u: ControlVec) maryam.EvalError!void {
            const prior_sigmas = try sigmaPoints(self.x, self.P);

            var propagated: [num_sigma]Core.StateVec = undefined;
            for (0..num_sigma) |i| propagated[i] = Model.f(prior_sigmas[i], u);

            var mean = Core.StateVec.zero();
            for (0..num_sigma) |i| {
                mean = Eq.WeightedAccumulate(Core.StateVec).eval(.{ .acc = mean, .w = wm[i], .point = propagated[i] });
            }

            var cov = Core.StateMat.zero();
            for (0..num_sigma) |i| {
                const diff = Eq.Deviation(Core.StateVec).eval(.{ .point = propagated[i], .mean = mean });
                cov = Eq.WeightedOuterAccumulate(Core.StateMat, Core.StateVec).eval(.{ .acc = cov, .w = wc[i], .diff = diff });
            }
            cov = Eq.AddProcessNoise.eval(.{ .cov = cov, .Q = self.Q });

            self.x = mean;
            self.P = cov;
            self.sigma_points = propagated;
        }

        pub fn update(self: *Self, z: Core.MeasureVec) maryam.EvalError!void {
            var measurement_sigmas: [num_sigma]Core.MeasureVec = undefined;
            for (0..num_sigma) |i| measurement_sigmas[i] = Model.h(self.sigma_points[i]);

            var z_mean = Core.MeasureVec.zero();
            for (0..num_sigma) |i| {
                z_mean = Eq.WeightedAccumulate(Core.MeasureVec).eval(.{ .acc = z_mean, .w = wm[i], .point = measurement_sigmas[i] });
            }

            var s = Core.MeasureNoise.zero();
            var pxz = CrossCov.zero();
            for (0..num_sigma) |i| {
                const z_diff = Eq.Deviation(Core.MeasureVec).eval(.{ .point = measurement_sigmas[i], .mean = z_mean });
                const x_diff = Eq.Deviation(Core.StateVec).eval(.{ .point = self.sigma_points[i], .mean = self.x });

                s = Eq.WeightedOuterAccumulate(Core.MeasureNoise, Core.MeasureVec).eval(.{ .acc = s, .w = wc[i], .diff = z_diff });
                pxz = Eq.CrossAccumulate.eval(.{ .acc = pxz, .w = wc[i], .x_diff = x_diff, .z_diff = z_diff });
            }
            s = Eq.AddMeasureNoise.eval(.{ .s = s, .R = self.R });

            const gain = try Eq.GainK.eval(.{ .S = s, .Pxz = pxz });

            const y = residual(z, z_mean);
            // "x + K @ y" doesn't reference H or F at all -- unlike every
            // other equation in kalman_core.KalmanCore, applying a gain to a
            // residual is pure shape (K: n x m, y: m x 1), regardless of
            // whether K came from a linear/Jacobian H or, as here,
            // sigma-point statistics. So this reuses Core.ApplyGain directly
            // instead of redeclaring an identical equation.
            self.x = Core.ApplyGain.eval(.{ .x = self.x, .K = gain, .y = y });
            const p_updated = Eq.UpdateP.eval(.{ .P = self.P, .K = gain, .S = s });
            self.P = Eq.Symmetrize.eval(.{ .P = p_updated });
            self.last_K = gain;
            self.last_y = y;
            self.last_S = s;
        }
    };
}

test "1D nonlinear-measurement filter matches a hand-derived closed form for h(x) = sin(x)" {
    const Vec1 = maryam.MatrixType(1, 1);

    const Model = struct {
        pub fn f(x: Vec1, u: Vec1) Vec1 {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn h(x: Vec1) Vec1 {
            var out = Vec1.zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
    };

    const UKF = UnscentedKalmanFilter(1, 1, 1, Model);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = v;
            return m;
        }
    }.of;

    var filter = UKF{
        .x = scalar(0),
        .P = scalar(1),
        .Q = scalar(0),
        .R = scalar(0.1),
    };

    // n=1: lambda = alpha^2*(n+kappa)-n = 0, spread = sqrt(n+lambda) = 1,
    // wm=[0, 0.5, 0.5], wc=[2, 0.5, 0.5]. Cholesky of P=1 is L=1, so prior
    // sigma points are {0, 0+1*1, 0-1*1} = {0, 1, -1}.
    //
    // predict with u=0: f is the identity here, so propagated = {0, 1, -1}.
    // mean = 0*0 + 0.5*1 + 0.5*(-1) = 0; cov = 2*0 + 0.5*1 + 0.5*1 + Q(0) = 1.
    filter.predict(scalar(0)) catch unreachable;
    try std.testing.expectApproxEqAbs(@as(f64, 0), filter.x.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1), filter.P.data[0][0], 1e-9);

    // update against z=0.5: measurement sigmas are {sin(0), sin(1), sin(-1)}
    // = {0, s, -s} where s = sin(1), which by symmetry gives z_mean=0,
    // S = s^2 + R, Pxz = s (independently re-derived below, not just
    // re-running the implementation's own arithmetic).
    const s = @sin(@as(f64, 1.0));
    const expected_s = s * s + 0.1;
    const expected_pxz = s;
    const expected_k = expected_pxz / expected_s;
    const expected_x = 0 + expected_k * (0.5 - 0);
    const expected_p = 1 - expected_k * expected_s * expected_k;

    try filter.update(scalar(0.5));
    try std.testing.expectApproxEqAbs(expected_x, filter.x.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(expected_p, filter.P.data[0][0], 1e-9);
}

test "update() leaves P exactly symmetric even with a genuinely coupled, off-diagonal covariance" {
    // A 2-state, 2-measurement model whose h mixes both state components
    // nonlinearly, so K @ S @ K^T has real off-diagonal terms -- P[0][1] and
    // P[1][0] are each the result of an independently-rounded dot product
    // (different row/column traversal order), which IEEE754 doesn't
    // guarantee to agree bit-for-bit even though they're mathematically the
    // same value. Without `Eq.Symmetrize`, this is exactly the scenario that
    // can leave `P` slightly asymmetric and make the next `predict()`'s
    // `choleskyMatrix` call (which requires an exactly symmetric input)
    // needlessly fragile.
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

    const Model = struct {
        pub fn f(x: Vec2, u: Vec2) Vec2 {
            _ = u;
            return x;
        }
        pub fn h(x: Vec2) Vec2 {
            return vec(x.data[0][0] + x.data[1][0], @sin(x.data[0][0] - x.data[1][0]));
        }
    };

    const UKF = UnscentedKalmanFilter(2, 2, 2, Model);

    var filter = UKF{
        .x = vec(0.3, -0.2),
        .P = blk: {
            var m = Mat2.zero();
            m.data[0][0] = 1.0;
            m.data[0][1] = 0.4;
            m.data[1][0] = 0.4;
            m.data[1][1] = 0.8;
            break :blk m;
        },
        .Q = Mat2.zero(),
        .R = blk: {
            var m = Mat2.zero();
            m.data[0][0] = 0.05;
            m.data[1][1] = 0.05;
            break :blk m;
        },
    };

    try filter.predict(vec(0, 0));
    try filter.update(vec(0.6, 0.1));

    try std.testing.expectEqual(filter.P.data[0][1], filter.P.data[1][0]);
}
