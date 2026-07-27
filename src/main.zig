const std = @import("std");
const Io = std.Io;

const Purifier = @import("Purifier");

pub fn main(init: std.process.Init) !void {
    // In order to do I/O operations need an `Io` instance.
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    try Purifier.imu_bench.run(stdout_writer, io);
}
