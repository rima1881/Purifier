const std = @import("std");
const Io = std.Io;

// Referencing these here (even though main() only calls imu_bench/
// kitti_bench directly) is what makes `zig build test` discover ctrv.zig's/
// gps_ins.zig's/bench_common.zig's/readme_examples.zig's own tests -- same
// mechanism as src/root.zig's `test { refAllDecls(@This()); }` below.
const ctrv = @import("ctrv.zig");
const gps_ins = @import("gps_ins.zig");
const bench_common = @import("bench_common.zig");
const imu_bench = @import("imu_bench.zig");
const kitti_bench = @import("kitti_bench.zig");
const readme_examples = @import("readme_examples.zig");

pub fn main(init: std.process.Init) !void {
    // In order to do I/O operations need an `Io` instance.
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    try imu_bench.run(stdout_writer, io);
    try stdout_writer.print("\n\n", .{});
    try kitti_bench.run(stdout_writer, io);
}

test {
    std.testing.refAllDecls(@This());
}
