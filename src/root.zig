//! By convention, root.zig is the root source file when making a package.
//!
//! This is the core library only: the two filter engines and their shared
//! covariance math. Everything demonstrating how to use them (benchmarks,
//! specific models like CTRV/bicycle, the README's code examples) lives
//! under examples/ instead, which depends on this module the same way an
//! external consumer would.
const std = @import("std");
const Io = std.Io;

pub const kalman_core = @import("kalman_core.zig");
pub const kalman = @import("kalman.zig");
pub const extended_kalman = @import("extended_kalman.zig");
pub const iterated_extended_kalman = @import("iterated_extended_kalman.zig");
pub const unscented_kalman = @import("unscented_kalman.zig");
pub const square_root_kalman = @import("square_root_kalman.zig");
pub const error_state_kalman = @import("error_state_kalman.zig");
pub const filter_union = @import("filter_union.zig");
pub const adaptive_kalman = @import("adaptive_kalman.zig");

// Re-exported so consumers of this package can build `StateVec`/`StateMat`/
// etc. values (`maryam.MatrixType(n, m)`) without also having to add maryam
// as their own separate dependency -- `KalmanFilter`/`ExtendedKalmanFilter`'s
// fields are typed in terms of it either way.
pub const maryam = @import("maryam");

// Without this, `zig build test` only discovers `test` blocks in this exact
// file -- nothing forces the compiler to analyze (and thus find tests in)
// files that are merely `@import`ed, so kalman.zig's/extended_kalman.zig's
// tests would silently never run.
test {
    std.testing.refAllDecls(@This());
}
