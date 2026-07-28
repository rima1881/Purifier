//! Shared finite-difference Jacobian check, used by both `ctrv.zig`'s and
//! `gps_ins.zig`'s tests: verifies an analytic Jacobian against a numerical
//! central-difference approximation, since a wrong sign in a hand-derived
//! Jacobian fails silently (the filter just diverges) rather than with a
//! compile or runtime error.

const std = @import("std");

pub fn expectJacobianMatchesFiniteDifference(
    comptime InVec: type,
    comptime OutVec: type,
    func: *const fn (InVec) OutVec,
    jac: anytype,
    x: InVec,
) !void {
    const eps = 1e-6;
    const analytic = jac(x);

    for (0..InVec.nrows) |j| {
        var x_plus = x;
        var x_minus = x;
        x_plus.data[j][0] += eps;
        x_minus.data[j][0] -= eps;

        const f_plus = func(x_plus);
        const f_minus = func(x_minus);

        for (0..OutVec.nrows) |i| {
            const numeric = (f_plus.data[i][0] - f_minus.data[i][0]) / (2 * eps);
            try std.testing.expectApproxEqAbs(numeric, analytic.data[i][j], 1e-4);
        }
    }
}
