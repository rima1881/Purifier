//! Shared report shape for the benchmark drivers (`imu_bench.zig`,
//! `kitti_bench.zig`): px/py/vx/vy RMSE + max|err|, plus speed and struct
//! size, printed the same way regardless of how differently each filter
//! variant is implemented internally, so results are directly comparable.

const std = @import("std");
const Io = std.Io;

// Not used by anything below -- this exists purely so readme_examples.zig
// has a second importer besides main.zig. See the comment above
// readme_examples.zig's import in main.zig for why that's needed: a file
// imported *only* by the refAllDecls root doesn't get its tests discovered
// by `zig build test` in this Zig version, but does once a second,
// non-root file also imports it. bench_common.zig, already imported by
// imu_bench.zig/kitti_bench.zig in addition to main.zig, is a convenient
// place for that second edge.
const readme_examples = @import("readme_examples.zig");
comptime {
    _ = readme_examples;
}

pub const BenchResult = struct {
    n_scored: usize,
    n_singular: usize,
    rmse: [4]f64,
    max_abs: [4]f64,
    avg_ns_per_cycle: f64,
    struct_bytes: usize,
};

/// The px/py/vx/vy error bookkeeping every `runXxx` in `imu_bench.zig`/
/// `kitti_bench.zig` used to duplicate inline: accumulate squared error and
/// max|error| per scored cycle, skip-count singular updates, and total the
/// predict+update timing, then turn all of that into a `BenchResult`.
pub const ErrorAccumulator = struct {
    sum_sq: [4]f64 = .{ 0, 0, 0, 0 },
    max_abs: [4]f64 = .{ 0, 0, 0, 0 },
    n_scored: usize = 0,
    n_singular: usize = 0,
    filter_ns: i96 = 0,

    pub fn recordSingular(self: *ErrorAccumulator) void {
        self.n_singular += 1;
    }

    pub fn record(self: *ErrorAccumulator, errs: [4]f64) void {
        for (0..4) |i| {
            self.sum_sq[i] += errs[i] * errs[i];
            self.max_abs[i] = @max(self.max_abs[i], @abs(errs[i]));
        }
        self.n_scored += 1;
    }

    pub fn finish(self: ErrorAccumulator, struct_bytes: usize) BenchResult {
        const n: f64 = @floatFromInt(self.n_scored);
        return .{
            .n_scored = self.n_scored,
            .n_singular = self.n_singular,
            .rmse = .{ @sqrt(self.sum_sq[0] / n), @sqrt(self.sum_sq[1] / n), @sqrt(self.sum_sq[2] / n), @sqrt(self.sum_sq[3] / n) },
            .max_abs = self.max_abs,
            .avg_ns_per_cycle = @as(f64, @floatFromInt(self.filter_ns)) / n,
            .struct_bytes = struct_bytes,
        };
    }
};

/// Calls `filt.predict(u)` whether `predict` returns `void` (every EKF) or
/// `maryam.EvalError!void` (UKF, since its Cholesky step can fail) --
/// letting `imu_bench.zig`/`kitti_bench.zig` share one loop body across both
/// filter kinds instead of a copy-pasted "UKF version" of every `runXxx`.
/// `@TypeOf(filt.predict(u))` doesn't actually call `predict` (types are
/// resolved without evaluating the expression), so this costs nothing beyond
/// the one real call below.
pub fn predictErr(filt: anytype, u: anytype) ?@import("maryam").EvalError {
    if (comptime @TypeOf(filt.predict(u)) == void) {
        filt.predict(u);
        return null;
    }
    filt.predict(u) catch |e| return e;
    return null;
}

/// Builds a filter value from shared (x, cov, Q, R), writing `cov` into
/// whichever covariance-representation field the filter type actually has:
/// `P` (`kalman.KalmanFilter`, `extended_kalman.ExtendedKalmanFilter`,
/// `unscented_kalman.UnscentedKalmanFilter`) or `L`
/// (`square_root_kalman.SquareRootKalmanFilter`'s Cholesky factor,
/// `P = L @ L^T`). Lets `imu_bench.zig`/`kitti_bench.zig` share one loop
/// across all four filter kinds instead of a fourth copy-pasted "SR-KF
/// version" of every `runXxx`. Callers are responsible for passing the
/// value each filter kind actually expects (e.g. a Cholesky factor, not a
/// raw covariance, for `SquareRootKalmanFilter`) -- this only picks the
/// field name, it doesn't convert between representations.
pub fn makeFilter(comptime Filter: type, x: anytype, cov: anytype, Q: anytype, R: anytype) Filter {
    var v: Filter = undefined;
    v.x = x;
    v.Q = Q;
    v.R = R;
    if (comptime @hasField(Filter, "P")) {
        v.P = cov;
    } else {
        v.L = cov;
    }
    return v;
}

/// The read-side counterpart of `makeFilter`: pulls whichever
/// covariance-representation field `filt` has back out, as `Cov` (the
/// caller already knows the concrete matrix type, e.g. `ctrv.StateMat`).
pub fn covOf(comptime Cov: type, filt: anytype) Cov {
    return if (comptime @hasField(@TypeOf(filt), "P")) filt.P else filt.L;
}

pub fn report(w: *Io.Writer, label: []const u8, r: BenchResult) !void {
    try w.print("-- {s} --\n", .{label});
    try w.print("scored {d} predict+update cycles, {d} singular-matrix updates skipped\n", .{ r.n_scored, r.n_singular });
    try w.print("RMSE      px={d:.4}  py={d:.4}  vx={d:.4}  vy={d:.4}\n", .{ r.rmse[0], r.rmse[1], r.rmse[2], r.rmse[3] });
    try w.print("max |err| px={d:.4}  py={d:.4}  vx={d:.4}  vy={d:.4}\n", .{ r.max_abs[0], r.max_abs[1], r.max_abs[2], r.max_abs[3] });
    try w.print("speed: {d:.0} ns/cycle avg -> {d:.0} cycles/sec; filter struct = {d} bytes\n\n", .{
        r.avg_ns_per_cycle, 1_000_000_000.0 / r.avg_ns_per_cycle, r.struct_bytes,
    });
}
