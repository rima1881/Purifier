//! Adaptive Kalman Filter: wraps *any one* of this package's other six
//! filter variants (see `filter_union.FilterKind`) and re-estimates `Q`
//! online from the innovation sequence, using the innovation-based adaptive
//! estimation (IAE) method (Mehra 1970/1972; the specific `Q` formula below
//! is Mohamed & Schwarz 1999, "Adaptive Kalman Filtering for INS/GPS",
//! Journal of Geodesy 73(4)).
//!
//! Every other filter in this package takes `Q` as a fixed field the caller
//! has to hand-pick (see the "untuned" vs. "different Q" columns in
//! Readme.md's benchmarks) -- this is the one exception, and it isn't tied
//! to the linear model specifically: the adaptation only ever needs a
//! step's residual `y` and the gain `K` that mapped it onto a state
//! correction, and *every* variant already computes both (Jacobian-based
//! for EKF/IEKF/ESKF, sigma-point-based for UKF, QR-based for SR-KF) and now
//! exposes them as `last_K`/`last_y` (see `kalman.zig`'s doc comment on
//! those fields). This filter doesn't re-derive any of that itself -- it
//! drives whichever variant is active through the `FilterKind` union,
//! reads back its `last_K`/`last_y`, and does the same re-estimation
//! regardless of which algorithm produced them.
//!
//! The idea: after each `update()`, the active variant's `last_y` is pushed
//! into a fixed-size ring buffer holding the most recent `window`
//! innovations. Once the buffer is full, every subsequent `update()`
//! recomputes
//!   Chat = (1/window) * sum_{j in window} y_j @ y_j^T   -- sample innovation covariance
//!   Q'   = K @ Chat @ K^T                                -- this step's last_K
//! and writes `Q'` into the active variant's own `Q` field, used by the
//! *next* `predict()`. `Chat` reflects how the filter is actually behaving
//! (large innovations -> the current `Q` is too small to explain the
//! observed spread, so the update pulls it up; small innovations pull it
//! back down), rather than a value chosen once, offline, before ever
//! running against real data.
//!
//! `Kind` (a type built by `filter_union.FilterKind(n, k, m, Model,
//! iekf_iterations)`), `window`, and `forgetting_factor` are all
//! **comptime** parameters, not struct fields -- which filter algorithm is
//! active, how much history to average `Q`'s estimate over, and how much to
//! trust a fresh estimate over the running one, are all algorithm-shape
//! choices, not per-instance data (same reasoning
//! `iterated_extended_kalman.zig` gives for its own `max_iterations`). `Q`
//! is still a required field on whichever concrete filter struct the caller
//! constructs `active` from: it's the seed value used for every `predict()`
//! until the ring buffer fills for the first time, and the starting point
//! the forgetting-factor blend below is computed relative to.
//!
//! `K @ Chat @ K^T` is a congruence transform of `Chat`, which is itself an
//! exact sum of `y @ y^T` outer products (each exactly symmetric, and exact
//! elementwise addition preserves that) -- so `Q'` is mathematically always
//! symmetric positive-semidefinite. But, like `unscented_kalman.zig`'s
//! `P - K @ S @ K^T`, the *computed* `K @ Chat @ K^T` isn't guaranteed to
//! come out bit-exactly symmetric: entries `(i, j)` and `(j, i)` are two
//! independently-rounded dot products through a different multiply chain.
//! `update()` symmetrizes the result the same way (`0.5 * (Q' + Q'^T)`) for
//! the same reason: a `Q` that drifts asymmetric is a `Q` that can make a
//! later `predict()`'s `F @ P @ F^T + Q` (or the nonlinear variants'
//! equivalent) quietly stop being exactly symmetric too.
//!
//! **Forgetting-factor blend and a diagonal floor/ceiling**, added after
//! Readme.md's own benchmarks found the raw formula above actively hurts
//! accuracy on real trajectories (see "Adaptive Kalman Filter: findings"):
//! a single window's sample covariance is a noisy estimate, and on a
//! trajectory with genuine turning/accelerating legs, some of that variance
//! is transient tracking error and CTRV-style linearization mismatch, not
//! process-noise mismatch -- the raw estimator can't tell the difference and
//! overwrites `Q` wholesale on every window-full `update()` regardless.
//! `Config.forgetting_factor` (`b`) blends the new estimate with the running
//! value instead of replacing it: `Q' = (1 - b) * Q + b * estimate`, `b = 1`
//! reproducing the original, undamped formula exactly (every hand-derived
//! test written before this parameter existed still passes unchanged at
//! `b = 1` for that reason). `q_floor`/`q_ceiling` (runtime `Core.StateVec`
//! fields, defaulting to `0`/`+inf` -- a no-op unless the caller opts in)
//! clamp `Q`'s diagonal after every blend -- a **diagonal-only**
//! approximation of the textbook "floor Q's eigenvalues" fix, not the real
//! thing (`maryam` has no general eigendecomposition), exact only when `Q`
//! is (or `Config.diagonal_only` forces it to stay) diagonal. Measured
//! effect: real, but the best-performing `b` values are the smallest ones
//! tried, i.e. the fix works by approaching "barely adapt at all" rather
//! than finding a genuinely better adaptive operating point -- see the
//! findings section for the actual numbers and why.
//!
//! **Four further options, gated behind `Config` fields so `Config{
//! .window = w }` alone still reproduces the original undamped behavior**:
//!
//!   - `Config.burn_in`: the first `burn_in` calls to `update()` don't get
//!     pushed into the innovation ring buffer at all. `P` (or `L`) starts
//!     wide and the state estimate hasn't converged yet right after
//!     construction, so the very first would-be window is the least
//!     representative one available -- burn-in exists to keep it out of the
//!     buffer entirely rather than let it poison an early `Q`/`R` estimate
//!     that then has to be blended back out.
//!   - `Config.diagonal_only`: discards the off-diagonal entries of each
//!     `Q`/`R` *estimate* before blending, so the stored value stays exactly
//!     diagonal for as long as it started that way. This is what makes
//!     `q_floor`/`q_ceiling` (and `r_floor`/`r_ceiling`, see below) an
//!     *exact* eigenvalue clamp instead of an approximation (a diagonal
//!     matrix's eigenvalues are its diagonal entries) -- important, because
//!     combining the floor/ceiling with anything but a very small
//!     `forgetting_factor` while *not* diagonal-only can leave the diagonal
//!     clamped down while off-diagonal terms stay large, producing an
//!     invalid (non-positive-semidefinite) matrix and outright divergence
//!     (measured on real data: RMSE in the millions -- see the findings
//!     section). The real cost: any legitimate correlation structure in the
//!     estimate (e.g. `ctrv.processNoise`'s cross terms between position and
//!     velocity) is discarded too, not just noise.
//!   - `Config.adapt_r`: also re-estimates `R`, using the same window's
//!     `Chat` via Mohamed & Schwarz's *joint* Q/R formula:
//!     `R' = Chat - (Sbar - R)`, where `Sbar` is the window-averaged
//!     innovation covariance (`S`, exposed as every variant's `last_S` --
//!     see `kalman.zig`'s doc comment) and `R` is the value already in use
//!     for the whole window (only ever changed at a window boundary, so
//!     it's constant across one window by construction). Blended and
//!     clamped (`r_floor`/`r_ceiling`) exactly like `Q`.
//!   - `Config.huber_threshold`: if set, each buffered innovation is
//!     downweighted before entering `Chat`/`Sbar` based on its Mahalanobis
//!     distance `d = sqrt(y^T @ S^-1 @ y)` -- weight `1` if `d <=
//!     huber_threshold`, `huber_threshold / d` otherwise (a standard Huber
//!     robust-loss weight, not a hard exclude/include gate). Meant to
//!     downweight exactly the turning/accelerating-transient residual
//!     spikes the module doc comment above blames for the raw formula's
//!     instability, without needing a full outlier-rejection test.
//!
//! All four default to off (`burn_in = 0`, `diagonal_only = false`,
//! `adapt_r = false`, `huber_threshold = null`), so existing callers that
//! only set `.window` see no behavior change.

const std = @import("std");
const maryam = @import("maryam");
const kalman_core = @import("kalman_core.zig");

pub const Config = struct {
    /// Number of past innovations averaged into each Q (and, if `adapt_r`,
    /// R) re-estimate. The one field with no default -- every caller has to
    /// pick this explicitly, same reasoning `window` always had as a
    /// mandatory comptime parameter before this became a config struct.
    window: usize,
    /// b in Q' = (1 - b) * Q + b * estimate (and, if adapt_r, the same for
    /// R). Must be in (0, 1]; 1 reproduces the original undamped formula.
    forgetting_factor: f64 = 1.0,
    /// Number of update() calls to skip before the ring buffer starts
    /// filling, letting P/the state estimate settle from their initial
    /// values first.
    burn_in: usize = 0,
    /// If true, off-diagonal entries of each Q/R estimate are discarded
    /// before blending, keeping Q/R exactly diagonal and making
    /// q_floor/q_ceiling (r_floor/r_ceiling) an exact eigenvalue clamp
    /// instead of an approximation.
    diagonal_only: bool = false,
    /// If true, R is also re-estimated online (Mohamed & Schwarz's joint
    /// Q/R IAE formula) using the same window, not just Q.
    adapt_r: bool = false,
    /// Huber-style downweighting threshold on each buffered innovation's
    /// Mahalanobis distance (sqrt(y^T @ S^-1 @ y)). null disables weighting
    /// (every buffered sample counted equally -- the original behavior).
    huber_threshold: ?f64 = null,
};

pub fn AdaptiveKalmanFilter(comptime n: usize, comptime k: usize, comptime m: usize, comptime Kind: type, comptime config: Config) type {
    // The default 1000-branch comptime quota is a budget shared across every
    // `maryam.Equation(...)` parsed while analyzing this function body, not
    // a per-call one -- more equations (the blend/floor/Huber/R-adaptation
    // machinery) on top of the original three pushed larger (e.g. n=5 CTRV)
    // instantiations over it.
    @setEvalBranchQuota(50000);

    if (config.window < 1) @compileError("AdaptiveKalmanFilter needs Config.window >= 1 (the number of past innovations the Q estimate is averaged over)");
    if (config.forgetting_factor <= 0 or config.forgetting_factor > 1) @compileError("AdaptiveKalmanFilter needs Config.forgetting_factor in (0, 1] (1 = the original, undamped Q' = estimate formula)");

    const Core = kalman_core.KalmanCore(n, m);
    const ControlVec = maryam.MatrixType(k, 1);

    // Sample innovation covariance, accumulated one window entry at a time
    // -- same "weighted accumulate" idiom `unscented_kalman.zig`'s
    // `WeightedOuterAccumulate` uses, except the weights are no longer
    // always the uniform `1/window` (see Config.huber_threshold).
    const AccumulateInnovationCov = maryam.Equation("acc + w * (y @ y^T)", struct {
        acc: Core.MeasureNoise,
        w: f64,
        y: Core.MeasureVec,
    });
    // Sbar, the window-averaged innovation covariance S itself -- only used
    // when Config.adapt_r is set, to recover R' = Chat - (Sbar - R).
    const AccumulateS = maryam.Equation("acc + w * S", struct {
        acc: Core.MeasureNoise,
        w: f64,
        S: Core.MeasureNoise,
    });
    // Mahalanobis distance squared, y^T @ S^-1 @ y -- only evaluated when
    // Config.huber_threshold is set. Produces a MatrixType(1, 1); callers
    // pull `.data[0][0]` out and sqrt it.
    const MahalanobisSq = maryam.Equation("y^T @ S^-1 @ y", struct { y: Core.MeasureVec, S: Core.MeasureNoise });

    // Q' = K @ Chat @ K^T -- Mohamed & Schwarz (1999)'s innovation-based Q
    // estimator: maps the *observed* innovation spread back into state
    // space through the active variant's own gain.
    const AdaptQ = maryam.Equation("K @ Chat @ K^T", struct { K: Core.GainMat, Chat: Core.MeasureNoise });

    // See the module doc comment above for why this is needed even though
    // Q' is symmetric in exact arithmetic.
    const Symmetrize = maryam.Equation("0.5 * (Q + Q^T)", struct { Q: Core.StateMat });
    const SymmetrizeR = maryam.Equation("0.5 * (Q + Q^T)", struct { Q: Core.MeasureNoise });

    // Forgetting-factor blend: Q' = (1 - b) * Q_old + b * Q_estimate. Split
    // into a reusable scale (used twice, once per side of the blend) and a
    // plain add, rather than one combined equation string -- keeps `1 - b`
    // as ordinary comptime-known Zig arithmetic instead of relying on the
    // equation DSL to parse a scalar sub-expression before a `*`.
    const ScaleQ = maryam.Equation("c * Q", struct { c: f64, Q: Core.StateMat });
    const AddQ = maryam.Equation("A + B", struct { A: Core.StateMat, B: Core.StateMat });
    // Same shapes, for R -- only used when Config.adapt_r is set.
    const ScaleR = maryam.Equation("c * Q", struct { c: f64, Q: Core.MeasureNoise });
    const AddR = maryam.Equation("A + B", struct { A: Core.MeasureNoise, B: Core.MeasureNoise });
    const SubR = maryam.Equation("A - B", struct { A: Core.MeasureNoise, B: Core.MeasureNoise });

    return struct {
        const Self = @This();

        // Which filter algorithm is actually driving predict()/update() --
        // see `filter_union.FilterKind`. Holds one of the other six filter
        // structs directly (constructed the same way it would be used
        // standalone), so its own `x`/`P`/`Q`/etc. are always the current,
        // authoritative state -- this wrapper adds no state of its own
        // beyond the innovation ring buffer below.
        active: Kind,

        // Per-state-component bounds applied to Q's diagonal after every
        // blend -- see the module doc comment for why this is a
        // diagonal-only approximation of a true eigenvalue floor/ceiling
        // unless `Config.diagonal_only` is also set, and why that's a
        // deliberate simplification, not an oversight. Default to
        // +-infinity in the unconstrained direction, i.e. a no-op unless
        // the caller opts in.
        q_floor: Core.StateVec = Core.StateVec.zero(),
        q_ceiling: Core.StateVec = blk: {
            var v = Core.StateVec.zero();
            for (0..n) |i| v.data[i][0] = std.math.inf(f64);
            break :blk v;
        },
        // Same idea, for R -- only consulted when `Config.adapt_r` is set.
        r_floor: Core.MeasureVec = Core.MeasureVec.zero(),
        r_ceiling: Core.MeasureVec = blk: {
            var v = Core.MeasureVec.zero();
            for (0..m) |i| v.data[i][0] = std.math.inf(f64);
            break :blk v;
        },

        // Ring buffers of the most recent `window` innovations and their
        // innovation covariances (the latter only ever read when
        // `Config.adapt_r` or `Config.huber_threshold` is set, but always
        // populated -- keeping the write side unconditional is simpler than
        // threading a second comptime-shaped buffer decision through), and
        // how many entries are populated so far (saturates at `window`).
        innovations: [config.window]Core.MeasureVec = undefined,
        innovation_s: [config.window]Core.MeasureNoise = undefined,
        count: usize = 0,
        cursor: usize = 0,
        // Total update() calls seen, including ones during burn-in --
        // distinct from `count` (which only counts buffered entries) so
        // burn-in and the ring buffer don't have to share one counter.
        total_updates: usize = 0,

        pub fn predict(self: *Self, u: ControlVec) maryam.EvalError!void {
            // predict()'s return type genuinely differs per variant (UKF and
            // SR-KF can fail -- they draw a Cholesky factor from P/L each
            // predict() -- the other four can't), so this dispatches
            // explicitly per tag rather than a uniform `inline else`, unlike
            // update() below.
            switch (self.active) {
                .linear => |*f| f.predict(u),
                .ekf => |*f| f.predict(u),
                .iekf => |*f| f.predict(u),
                .ukf => |*f| try f.predict(u),
                .srkf => |*f| try f.predict(u),
                .eskf => |*f| f.predict(u),
            }
        }

        pub fn update(self: *Self, z: Core.MeasureVec) maryam.EvalError!void {
            // Every variant's update() returns the same maryam.EvalError!void,
            // so this (unlike predict() above) can dispatch generically.
            switch (self.active) {
                inline else => |*f| try f.update(z),
            }

            self.total_updates += 1;
            if (self.total_updates <= config.burn_in) return;

            const y = switch (self.active) {
                inline else => |*f| f.last_y,
            };
            const s = switch (self.active) {
                inline else => |*f| f.last_S,
            };

            self.innovations[self.cursor] = y;
            self.innovation_s[self.cursor] = s;
            self.cursor = (self.cursor + 1) % config.window;
            if (self.count < config.window) self.count += 1;

            if (self.count == config.window) {
                // Huber weights (or uniform 1/window if huber_threshold is
                // unset), normalized to sum to 1 so Chat/Sbar stay proper
                // weighted averages regardless of how many entries got
                // downweighted.
                var weights: [config.window]f64 = undefined;
                var weight_sum: f64 = 0;
                for (0..config.window) |i| {
                    const w: f64 = if (config.huber_threshold) |c| blk: {
                        const d2 = (try MahalanobisSq.eval(.{ .y = self.innovations[i], .S = self.innovation_s[i] })).data[0][0];
                        const d = @sqrt(@max(d2, 0));
                        break :blk if (d <= c) 1.0 else c / d;
                    } else 1.0;
                    weights[i] = w;
                    weight_sum += w;
                }

                var chat = Core.MeasureNoise.zero();
                var sbar = Core.MeasureNoise.zero();
                for (0..config.window) |i| {
                    const wn = weights[i] / weight_sum;
                    chat = AccumulateInnovationCov.eval(.{ .acc = chat, .w = wn, .y = self.innovations[i] });
                    if (config.adapt_r) {
                        sbar = AccumulateS.eval(.{ .acc = sbar, .w = wn, .S = self.innovation_s[i] });
                    }
                }

                const active_k = switch (self.active) {
                    inline else => |*f| f.last_K,
                };
                const q_raw = AdaptQ.eval(.{ .K = active_k, .Chat = chat });
                var q_estimate = Symmetrize.eval(.{ .Q = q_raw });
                if (config.diagonal_only) {
                    for (0..n) |i| for (0..n) |j| {
                        if (i != j) q_estimate.data[i][j] = 0;
                    };
                }

                const q_old = switch (self.active) {
                    inline else => |*f| f.Q,
                };
                var q_new = AddQ.eval(.{
                    .A = ScaleQ.eval(.{ .c = 1.0 - config.forgetting_factor, .Q = q_old }),
                    .B = ScaleQ.eval(.{ .c = config.forgetting_factor, .Q = q_estimate }),
                });
                for (0..n) |i| {
                    if (q_new.data[i][i] < self.q_floor.data[i][0]) q_new.data[i][i] = self.q_floor.data[i][0];
                    if (q_new.data[i][i] > self.q_ceiling.data[i][0]) q_new.data[i][i] = self.q_ceiling.data[i][0];
                }

                switch (self.active) {
                    inline else => |*f| f.Q = q_new,
                }

                if (config.adapt_r) {
                    const r_old = switch (self.active) {
                        inline else => |*f| f.R,
                    };
                    // R' = Chat - (Sbar - R): Sbar - R recovers the
                    // window-averaged H @ P @ H^T (R is constant across one
                    // window by construction -- it's only ever changed here,
                    // at a window boundary), and subtracting that back out
                    // of Chat isolates the measurement-noise contribution.
                    const hpht = SubR.eval(.{ .A = sbar, .B = r_old });
                    const r_raw = SubR.eval(.{ .A = chat, .B = hpht });
                    var r_estimate = SymmetrizeR.eval(.{ .Q = r_raw });
                    if (config.diagonal_only) {
                        for (0..m) |i| for (0..m) |j| {
                            if (i != j) r_estimate.data[i][j] = 0;
                        };
                    }

                    var r_new = AddR.eval(.{
                        .A = ScaleR.eval(.{ .c = 1.0 - config.forgetting_factor, .Q = r_old }),
                        .B = ScaleR.eval(.{ .c = config.forgetting_factor, .Q = r_estimate }),
                    });
                    for (0..m) |i| {
                        if (r_new.data[i][i] < self.r_floor.data[i][0]) r_new.data[i][i] = self.r_floor.data[i][0];
                        if (r_new.data[i][i] > self.r_ceiling.data[i][0]) r_new.data[i][i] = self.r_ceiling.data[i][0];
                    }

                    switch (self.active) {
                        inline else => |*f| f.R = r_new,
                    }
                }
            }
        }
    };
}

const filter_union = @import("filter_union.zig");
const kalman = @import("kalman.zig");
const extended_kalman = @import("extended_kalman.zig");
const unscented_kalman = @import("unscented_kalman.zig");

test "linear variant, window=1: Q adapts to exactly (x_new - x_pred)^2" {
    // With window=1, Chat collapses to the single latest y @ y^T (no
    // averaging), so Q' = K @ (y y^T) @ K^T = (K y)(K y)^T -- and K @ y is
    // exactly the state correction ApplyGain just added (x_new - x_pred).
    // In this 1D case that's a plain scalar square, checkable directly
    // against the filter's own x, with no separate K/y bookkeeping needed.
    //
    // Model is only ever exercised through the `.linear` tag below (which
    // doesn't reference Model at all), but `predict()`/`update()` switch on
    // a *runtime* tag, so every arm -- including the five nonlinear ones --
    // has to compile regardless (see filter_union.zig's doc comment). A
    // real, if unused here, EKF-style Model is required, not a placeholder.
    const Model = struct {
        pub fn f(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            _ = x;
            _ = u;
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = maryam.MatrixType(1, 1).zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };
    const Kind = filter_union.FilterKind(1, 1, 1, Model, 1);
    const AKF = AdaptiveKalmanFilter(1, 1, 1, Kind, .{ .window = 1 });
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var mtx = Vec1.zero();
            mtx.data[0][0] = v;
            return mtx;
        }
    }.of;

    var filter = AKF{
        .active = .{ .linear = kalman.KalmanFilter(1, 1, 1){
            .x = scalar(0),
            .P = scalar(1),
            .F = scalar(1),
            .B = scalar(0),
            .Q = scalar(0), // seed -- overwritten after the first update(), since window=1
            .H = scalar(1),
            .R = scalar(1),
        } },
    };

    try filter.predict(scalar(0)); // x_pred = 0, P_pred = 1
    const x_pred = filter.active.linear.x.data[0][0];

    try filter.update(scalar(2)); // z=2: S=2, K=0.5, y=2, x_new = 0 + 0.5*2 = 1
    try std.testing.expectApproxEqAbs(@as(f64, 1), filter.active.linear.x.data[0][0], 1e-9);

    const correction = filter.active.linear.x.data[0][0] - x_pred;
    const expected_q = correction * correction; // (1 - 0)^2 = 1
    try std.testing.expectApproxEqAbs(expected_q, filter.active.linear.Q.data[0][0], 1e-9);
}

test "linear variant, window=3: Q matches a hand-derived value once the ring buffer fills" {
    // Same scenario, formula, and hand-derived expected values as this
    // filter's original (pre-union) test: 1D system, F=1, H=1, B=0, Q seed
    // = 0, R = 1, window = 3, fed z = [2, 4, 6]. Q stays at its seed (0) for
    // the whole run -- it's only overwritten *after* the 3rd update(), too
    // late to affect any predict() in this test -- so every P/K/x value is
    // the plain, deterministic Joseph-form recursion with a fixed Q=0:
    //   step 1: P_pred=1, S=2, K1=0.5,  y1=2-0=2, x1=0+0.5*2=1,     P1=0.5
    //   step 2: P_pred=0.5, S=1.5, K2=1/3, y2=4-1=3, x2=1+1/3*3=2,  P2=1/3
    //   step 3: P_pred=1/3, S=4/3, K3=1/4, y3=6-2=4, x3=2+1/4*4=3,  P3=1/4
    // Chat = (2^2 + 3^2 + 4^2) / 3 = 29/3
    // Q'   = K3^2 * Chat = (1/4)^2 * 29/3 = 29/48
    //
    // Model (see the previous test) is only ever exercised through the
    // `.linear` tag, but has to satisfy the full EKF interface anyway --
    // see that test's comment on why.
    const Model = struct {
        pub fn f(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            _ = x;
            _ = u;
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = maryam.MatrixType(1, 1).zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };
    const Kind = filter_union.FilterKind(1, 1, 1, Model, 1);
    const AKF = AdaptiveKalmanFilter(1, 1, 1, Kind, .{ .window = 3 });
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var mtx = Vec1.zero();
            mtx.data[0][0] = v;
            return mtx;
        }
    }.of;

    var filter = AKF{
        .active = .{ .linear = kalman.KalmanFilter(1, 1, 1){
            .x = scalar(0),
            .P = scalar(1),
            .F = scalar(1),
            .B = scalar(0),
            .Q = scalar(0),
            .H = scalar(1),
            .R = scalar(1),
        } },
    };

    for ([_]f64{ 2, 4, 6 }) |z| {
        try filter.predict(scalar(0));
        try filter.update(scalar(z));
    }

    try std.testing.expectApproxEqAbs(@as(f64, 3), filter.active.linear.x.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), filter.active.linear.P.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 29.0 / 48.0), filter.active.linear.Q.data[0][0], 1e-9);
}

test "ekf variant: the same adaptation mechanism works through a genuinely nonlinear model" {
    // Confirms this isn't special-cased to the linear filter: same
    // h(x) = sin(x) model every other file in this package tests the EKF
    // with, wrapped in AdaptiveKalmanFilter via the .ekf tag instead of
    // being driven standalone. A badly-underestimated seed Q should still
    // self-correct, exactly as it does for the linear variant above.
    const Model = struct {
        pub fn f(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            _ = x;
            _ = u;
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = maryam.MatrixType(1, 1).zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };

    const Kind = filter_union.FilterKind(1, 1, 1, Model, 1);
    const AKF = AdaptiveKalmanFilter(1, 1, 1, Kind, .{ .window = 5 });
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var mtx = Vec1.zero();
            mtx.data[0][0] = v;
            return mtx;
        }
    }.of;

    var filter = AKF{
        .active = .{ .ekf = extended_kalman.ExtendedKalmanFilter(1, 1, 1, Model){
            .x = scalar(0),
            .P = scalar(1),
            .Q = scalar(1e-6), // badly underestimated seed
            .R = scalar(0.1),
        } },
    };
    const seed_q = filter.active.ekf.Q.data[0][0];

    // Alternating measurements far from what a static-ish state near 0
    // could explain via sin(x) in [-1, 1] with Q=1e-6 -- a sustained,
    // genuinely large innovation spread.
    var i: usize = 0;
    while (i < 20) : (i += 1) {
        try filter.predict(scalar(0));
        const z: f64 = if (i % 2 == 0) 0.9 else -0.9;
        try filter.update(scalar(z));
    }

    try std.testing.expect(filter.active.ekf.Q.data[0][0] > seed_q * 100);
}

test "ukf variant: Q stays exactly symmetric with sigma-point-derived last_K" {
    // Same genuinely-coupled 2-state model unscented_kalman.zig's own
    // symmetry test uses, driven here through the .ukf tag -- proves the
    // sigma-point gain (last_K = the UKF's own cross-covariance-derived K,
    // not a Jacobian) plugs into the same Symmetrize-guarded formula without
    // any UKF-specific handling in this file.
    //
    // jacobianF/jacobianH are never called by the .ukf tag itself (UKF only
    // needs f/h), but predict()/update() switch on a *runtime* tag, so the
    // other five (Jacobian-needing) arms still have to compile -- see
    // filter_union.zig's doc comment. Real derivatives are supplied anyway
    // for clarity, even though this test never exercises them.
    const Model = struct {
        pub fn f(x: maryam.MatrixType(2, 1), u: maryam.MatrixType(2, 1)) maryam.MatrixType(2, 1) {
            _ = u;
            return x;
        }
        pub fn jacobianF(x: maryam.MatrixType(2, 1), u: maryam.MatrixType(2, 1)) maryam.MatrixType(2, 2) {
            _ = x;
            _ = u;
            var m = maryam.MatrixType(2, 2).zero();
            m.data[0][0] = 1;
            m.data[1][1] = 1;
            return m;
        }
        pub fn h(x: maryam.MatrixType(2, 1)) maryam.MatrixType(2, 1) {
            var out = maryam.MatrixType(2, 1).zero();
            out.data[0][0] = x.data[0][0] + x.data[1][0];
            out.data[1][0] = @sin(x.data[0][0] - x.data[1][0]);
            return out;
        }
        pub fn jacobianH(x: maryam.MatrixType(2, 1)) maryam.MatrixType(2, 2) {
            const c = @cos(x.data[0][0] - x.data[1][0]);
            var m = maryam.MatrixType(2, 2).zero();
            m.data[0][0] = 1;
            m.data[0][1] = 1;
            m.data[1][0] = c;
            m.data[1][1] = -c;
            return m;
        }
    };

    const Kind = filter_union.FilterKind(2, 2, 2, Model, 1);
    const AKF = AdaptiveKalmanFilter(2, 2, 2, Kind, .{ .window = 4 });
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

    var filter = AKF{
        .active = .{ .ukf = unscented_kalman.UnscentedKalmanFilter(2, 2, 2, Model){
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
        } },
    };

    const zs = [_]Vec2{ vec(0.6, 0.1), vec(0.5, -0.2), vec(0.4, 0.3), vec(0.55, -0.1) };
    for (zs) |z| {
        try filter.predict(vec(0, 0));
        try filter.update(z);
    }

    try std.testing.expectEqual(filter.active.ukf.Q.data[0][1], filter.active.ukf.Q.data[1][0]);
    try std.testing.expect(filter.active.ukf.Q.data[0][0] > 0);
    try std.testing.expect(filter.active.ukf.Q.data[1][1] > 0);
}

test "forgetting_factor < 1 blends the estimate with the running Q instead of replacing it" {
    // Identical scenario to "linear variant, window=3" above (same model,
    // same z = [2, 4, 6], same Q seed = 0, same hand-derived Q estimate =
    // 29/48 at the 3rd update()) -- x/P are untouched by this change (Q
    // only ever affects a *later* predict(), too late to matter within this
    // 3-step run), so only the final Q differs: with forgetting_factor=0.5,
    // Q' = (1 - 0.5) * 0 + 0.5 * (29/48) = 29/96.
    const Model = struct {
        pub fn f(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            _ = x;
            _ = u;
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = maryam.MatrixType(1, 1).zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };
    const Kind = filter_union.FilterKind(1, 1, 1, Model, 1);
    const AKF = AdaptiveKalmanFilter(1, 1, 1, Kind, .{ .window = 3, .forgetting_factor = 0.5 });
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var mtx = Vec1.zero();
            mtx.data[0][0] = v;
            return mtx;
        }
    }.of;

    var filter = AKF{
        .active = .{ .linear = kalman.KalmanFilter(1, 1, 1){
            .x = scalar(0),
            .P = scalar(1),
            .F = scalar(1),
            .B = scalar(0),
            .Q = scalar(0),
            .H = scalar(1),
            .R = scalar(1),
        } },
    };

    for ([_]f64{ 2, 4, 6 }) |z| {
        try filter.predict(scalar(0));
        try filter.update(scalar(z));
    }

    try std.testing.expectApproxEqAbs(@as(f64, 3), filter.active.linear.x.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), filter.active.linear.P.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 29.0 / 96.0), filter.active.linear.Q.data[0][0], 1e-9);
}

test "q_ceiling clamps a blended Q that would otherwise exceed it" {
    // Same window=1 scenario as "linear variant, window=1" above (K=0.5,
    // y=2, correction=1, so the unclamped estimate would be exactly 1) --
    // q_ceiling=0.3 should clamp the stored Q down to exactly 0.3, not 1.
    const Model = struct {
        pub fn f(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            _ = x;
            _ = u;
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = maryam.MatrixType(1, 1).zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };
    const Kind = filter_union.FilterKind(1, 1, 1, Model, 1);
    const AKF = AdaptiveKalmanFilter(1, 1, 1, Kind, .{ .window = 1 });
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var mtx = Vec1.zero();
            mtx.data[0][0] = v;
            return mtx;
        }
    }.of;

    var filter = AKF{
        .active = .{ .linear = kalman.KalmanFilter(1, 1, 1){
            .x = scalar(0),
            .P = scalar(1),
            .F = scalar(1),
            .B = scalar(0),
            .Q = scalar(0),
            .H = scalar(1),
            .R = scalar(1),
        } },
        .q_ceiling = scalar(0.3),
    };

    try filter.predict(scalar(0));
    try filter.update(scalar(2));

    try std.testing.expectApproxEqAbs(@as(f64, 0.3), filter.active.linear.Q.data[0][0], 1e-9);
}

test "q_floor clamps a blended Q that would otherwise fall below it" {
    // Same shape as the ceiling test, but with R large enough (99) that the
    // gain (K = P/(P+R) = 1/100) and thus the correction (K*y = 0.02) are
    // both tiny, making the unclamped estimate (0.02^2 = 0.0004) fall well
    // below q_floor=0.01.
    const Model = struct {
        pub fn f(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            _ = x;
            _ = u;
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = maryam.MatrixType(1, 1).zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };
    const Kind = filter_union.FilterKind(1, 1, 1, Model, 1);
    const AKF = AdaptiveKalmanFilter(1, 1, 1, Kind, .{ .window = 1 });
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var mtx = Vec1.zero();
            mtx.data[0][0] = v;
            return mtx;
        }
    }.of;

    var filter = AKF{
        .active = .{ .linear = kalman.KalmanFilter(1, 1, 1){
            .x = scalar(0),
            .P = scalar(1),
            .F = scalar(1),
            .B = scalar(0),
            .Q = scalar(0),
            .H = scalar(1),
            .R = scalar(99),
        } },
        .q_floor = scalar(0.01),
    };

    try filter.predict(scalar(0));
    try filter.update(scalar(2));

    try std.testing.expectApproxEqAbs(@as(f64, 0.01), filter.active.linear.Q.data[0][0], 1e-9);
}

test "burn_in delays the ring buffer, not the underlying filtering" {
    // 1D system, F=1, H=1, Q seed=0, R=1, window=3, burn_in=2, fed
    // z=[2,4,6,8,10]. predict()/update() run their ordinary math on every
    // one of the 5 steps regardless (burn_in only gates the ring buffer),
    // so x/P/K/y follow the plain Joseph-form recursion the whole way:
    //   step 1: y=2,  S=2,   K=1/2, x=1, P=1/2   (excluded: burn_in=2)
    //   step 2: y=3,  S=3/2, K=1/3, x=2, P=1/3   (excluded: burn_in=2)
    //   step 3: y=4,  S=4/3, K=1/4, x=3, P=1/4   (buffered #1)
    //   step 4: y=5,  S=5/4, K=1/5, x=4, P=1/5   (buffered #2)
    //   step 5: y=6,  S=6/5, K=1/6, x=5, P=1/6   (buffered #3 -- window fills)
    // Chat = (y3^2 + y4^2 + y5^2) / 3 = (16 + 25 + 36) / 3 = 77/3
    // Q'   = K5^2 * Chat = (1/6)^2 * 77/3 = 77/108
    // If burn_in didn't work, the window would instead fill at step 3 from
    // y1/y2/y3, giving a different (and smaller) Q.
    const Model = struct {
        pub fn f(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            _ = x;
            _ = u;
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = maryam.MatrixType(1, 1).zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };
    const Kind = filter_union.FilterKind(1, 1, 1, Model, 1);
    const AKF = AdaptiveKalmanFilter(1, 1, 1, Kind, .{ .window = 3, .burn_in = 2 });
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var mtx = Vec1.zero();
            mtx.data[0][0] = v;
            return mtx;
        }
    }.of;

    var filter = AKF{
        .active = .{ .linear = kalman.KalmanFilter(1, 1, 1){
            .x = scalar(0),
            .P = scalar(1),
            .F = scalar(1),
            .B = scalar(0),
            .Q = scalar(0),
            .H = scalar(1),
            .R = scalar(1),
        } },
    };

    for ([_]f64{ 2, 4, 6, 8, 10 }) |z| {
        try filter.predict(scalar(0));
        try filter.update(scalar(z));
    }

    try std.testing.expectApproxEqAbs(@as(f64, 5), filter.active.linear.x.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 6.0), filter.active.linear.P.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 77.0 / 108.0), filter.active.linear.Q.data[0][0], 1e-9);
}

test "diagonal_only keeps Q exactly diagonal even when the raw estimate has real off-diagonal terms" {
    // Same genuinely-coupled 2-state UKF model unscented_kalman.zig's own
    // symmetry test uses (see "ukf variant: Q stays exactly symmetric..."
    // above), which without diagonal_only produces a Q with real nonzero
    // off-diagonal entries -- diagonal_only should discard them entirely,
    // not just make them symmetric.
    const Model = struct {
        pub fn f(x: maryam.MatrixType(2, 1), u: maryam.MatrixType(2, 1)) maryam.MatrixType(2, 1) {
            _ = u;
            return x;
        }
        pub fn jacobianF(x: maryam.MatrixType(2, 1), u: maryam.MatrixType(2, 1)) maryam.MatrixType(2, 2) {
            _ = x;
            _ = u;
            var m = maryam.MatrixType(2, 2).zero();
            m.data[0][0] = 1;
            m.data[1][1] = 1;
            return m;
        }
        pub fn h(x: maryam.MatrixType(2, 1)) maryam.MatrixType(2, 1) {
            var out = maryam.MatrixType(2, 1).zero();
            out.data[0][0] = x.data[0][0] + x.data[1][0];
            out.data[1][0] = @sin(x.data[0][0] - x.data[1][0]);
            return out;
        }
        pub fn jacobianH(x: maryam.MatrixType(2, 1)) maryam.MatrixType(2, 2) {
            const c = @cos(x.data[0][0] - x.data[1][0]);
            var m = maryam.MatrixType(2, 2).zero();
            m.data[0][0] = 1;
            m.data[0][1] = 1;
            m.data[1][0] = c;
            m.data[1][1] = -c;
            return m;
        }
    };

    const Kind = filter_union.FilterKind(2, 2, 2, Model, 1);
    const AKF = AdaptiveKalmanFilter(2, 2, 2, Kind, .{ .window = 4, .diagonal_only = true });
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

    var filter = AKF{
        .active = .{ .ukf = unscented_kalman.UnscentedKalmanFilter(2, 2, 2, Model){
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
        } },
    };

    const zs = [_]Vec2{ vec(0.6, 0.1), vec(0.5, -0.2), vec(0.4, 0.3), vec(0.55, -0.1) };
    for (zs) |z| {
        try filter.predict(vec(0, 0));
        try filter.update(z);
    }

    try std.testing.expectEqual(@as(f64, 0), filter.active.ukf.Q.data[0][1]);
    try std.testing.expectEqual(@as(f64, 0), filter.active.ukf.Q.data[1][0]);
    try std.testing.expect(filter.active.ukf.Q.data[0][0] > 0);
    try std.testing.expect(filter.active.ukf.Q.data[1][1] > 0);
}

test "adapt_r re-estimates R using Mohamed & Schwarz's joint Q/R formula, hand-derived" {
    // Same 1D scenario and same Q result as "linear variant, window=3"
    // above (F=1, H=1, Q seed=0, R seed=1, window=3, z=[2,4,6]) -- Q
    // adaptation is unaffected by adapt_r (Q'=29/48 either way). R is
    // additionally re-estimated: with R constant at its seed (1) for the
    // whole window (it's only overwritten after the 3rd update()),
    //   S = [2, 3/2, 4/3], Chat = 29/3 (same as before)
    //   Sbar = mean(S) = 29/18
    //   hpht = Sbar - R = 29/18 - 1 = 11/18
    //   R'   = Chat - hpht = 29/3 - 11/18 = 163/18
    const Model = struct {
        pub fn f(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            _ = x;
            _ = u;
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = maryam.MatrixType(1, 1).zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };
    const Kind = filter_union.FilterKind(1, 1, 1, Model, 1);
    const AKF = AdaptiveKalmanFilter(1, 1, 1, Kind, .{ .window = 3, .adapt_r = true });
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var mtx = Vec1.zero();
            mtx.data[0][0] = v;
            return mtx;
        }
    }.of;

    var filter = AKF{
        .active = .{ .linear = kalman.KalmanFilter(1, 1, 1){
            .x = scalar(0),
            .P = scalar(1),
            .F = scalar(1),
            .B = scalar(0),
            .Q = scalar(0),
            .H = scalar(1),
            .R = scalar(1),
        } },
    };

    for ([_]f64{ 2, 4, 6 }) |z| {
        try filter.predict(scalar(0));
        try filter.update(scalar(z));
    }

    try std.testing.expectApproxEqAbs(@as(f64, 29.0 / 48.0), filter.active.linear.Q.data[0][0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 163.0 / 18.0), filter.active.linear.R.data[0][0], 1e-9);
}

test "huber_threshold downweights a high-Mahalanobis-distance innovation instead of counting it equally" {
    // Same 1D scenario as "linear variant, window=3" above (z=[2,4,6]),
    // whose 3 innovations have increasing Mahalanobis distances
    // (sqrt(y^2/S) = sqrt(2), sqrt(6), sqrt(12) ~= 1.414, 2.449, 3.464).
    // huber_threshold=2.0 leaves the first at full weight and downweights
    // the other two (weight = threshold/distance), independently computed
    // (not re-run from the implementation's own arithmetic):
    //   normalized weights ~= 0.41774, 0.34108, 0.24118
    //   Chat ~= 8.599578345780296  (vs. 29/3 ~= 9.667 unweighted)
    //   Q'   = K3^2 * Chat = (1/4)^2 * 8.599578345780296 ~= 0.5374736466
    const Model = struct {
        pub fn f(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: maryam.MatrixType(1, 1), u: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            _ = x;
            _ = u;
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var out = maryam.MatrixType(1, 1).zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: maryam.MatrixType(1, 1)) maryam.MatrixType(1, 1) {
            var m = maryam.MatrixType(1, 1).zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };
    const Kind = filter_union.FilterKind(1, 1, 1, Model, 1);
    const AKF = AdaptiveKalmanFilter(1, 1, 1, Kind, .{ .window = 3, .huber_threshold = 2.0 });
    const Vec1 = maryam.MatrixType(1, 1);

    const scalar = struct {
        fn of(v: f64) Vec1 {
            var mtx = Vec1.zero();
            mtx.data[0][0] = v;
            return mtx;
        }
    }.of;

    var filter = AKF{
        .active = .{ .linear = kalman.KalmanFilter(1, 1, 1){
            .x = scalar(0),
            .P = scalar(1),
            .F = scalar(1),
            .B = scalar(0),
            .Q = scalar(0),
            .H = scalar(1),
            .R = scalar(1),
        } },
    };

    for ([_]f64{ 2, 4, 6 }) |z| {
        try filter.predict(scalar(0));
        try filter.update(scalar(z));
    }

    try std.testing.expectApproxEqAbs(@as(f64, 0.5374736466112685), filter.active.linear.Q.data[0][0], 1e-9);
    // And the actual point: weighting down the highest-distance innovations
    // pulls Q below what the unweighted formula would have produced.
    try std.testing.expect(filter.active.linear.Q.data[0][0] < 29.0 / 48.0);
}
