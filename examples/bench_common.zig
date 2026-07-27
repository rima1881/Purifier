//! Shared report shape for the benchmark drivers (`imu_bench.zig`,
//! `kitti_bench.zig`): px/py/vx/vy RMSE + max|err|, plus speed and struct
//! size, printed the same way regardless of how differently each filter
//! variant is implemented internally, so results are directly comparable.

const std = @import("std");
const Io = std.Io;

pub const BenchResult = struct {
    n_scored: usize,
    n_singular: usize,
    rmse: [4]f64,
    max_abs: [4]f64,
    avg_ns_per_cycle: f64,
    struct_bytes: usize,
};

pub fn report(w: *Io.Writer, label: []const u8, r: BenchResult) !void {
    try w.print("-- {s} --\n", .{label});
    try w.print("scored {d} predict+update cycles, {d} singular-matrix updates skipped\n", .{ r.n_scored, r.n_singular });
    try w.print("RMSE      px={d:.4}  py={d:.4}  vx={d:.4}  vy={d:.4}\n", .{ r.rmse[0], r.rmse[1], r.rmse[2], r.rmse[3] });
    try w.print("max |err| px={d:.4}  py={d:.4}  vx={d:.4}  vy={d:.4}\n", .{ r.max_abs[0], r.max_abs[1], r.max_abs[2], r.max_abs[3] });
    try w.print("speed: {d:.0} ns/cycle avg -> {d:.0} cycles/sec; filter struct = {d} bytes\n\n", .{
        r.avg_ns_per_cycle, 1_000_000_000.0 / r.avg_ns_per_cycle, r.struct_bytes,
    });
}
