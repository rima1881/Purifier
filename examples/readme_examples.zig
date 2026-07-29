//! The exact code shown in Readme.md's "Using this as a library" section,
//! kept here as tests so the docs can't silently drift out of sync with the
//! actual API.

const std = @import("std");
const Purifier = @import("Purifier");
const kalman = Purifier.kalman;
const extended_kalman = Purifier.extended_kalman;
const iterated_extended_kalman = Purifier.iterated_extended_kalman;
const unscented_kalman = Purifier.unscented_kalman;
const square_root_kalman = Purifier.square_root_kalman;
const error_state_kalman = Purifier.error_state_kalman;
const filter_union = Purifier.filter_union;
const adaptive_kalman = Purifier.adaptive_kalman;
const maryam = @import("maryam");

test "Readme.md: linear KalmanFilter example" {
    // n=2 state [position, velocity], k=1 control (unused here), m=1
    // measurement (position only).
    const KF = kalman.KalmanFilter(2, 1, 1);
    const StateVec = maryam.MatrixType(2, 1);
    const StateMat = maryam.MatrixType(2, 2);
    const ControlVec = maryam.MatrixType(1, 1);
    const ControlMat = maryam.MatrixType(2, 1);
    const MeasureVec = maryam.MatrixType(1, 1);
    const MeasureMat = maryam.MatrixType(1, 2);
    const MeasureNoise = maryam.MatrixType(1, 1);

    var filter = KF{
        .x = StateVec.zero(), // start at position=0, velocity=0
        .P = blk: {
            var m = StateMat.zero();
            m.data[0][0] = 1;
            m.data[1][1] = 1;
            break :blk m;
        },
        .F = blk: { // constant-velocity model, dt=1: position += velocity
            var m = StateMat.zero();
            m.data[0][0] = 1;
            m.data[0][1] = 1;
            m.data[1][1] = 1;
            break :blk m;
        },
        .B = ControlMat.zero(), // no control input in this example
        .Q = StateMat.zero(),
        .H = blk: { // measure position only
            var m = MeasureMat.zero();
            m.data[0][0] = 1;
            break :blk m;
        },
        .R = blk: {
            var m = MeasureNoise.zero();
            m.data[0][0] = 0.1;
            break :blk m;
        },
    };

    filter.predict(ControlVec.zero());

    var z = MeasureVec.zero();
    z.data[0][0] = 1.0; // measured position = 1.0
    try filter.update(z);

    try std.testing.expect(filter.x.data[0][0] > 0); // moved toward the measurement
}

test "Readme.md: ExtendedKalmanFilter example" {
    // 1-state EKF estimating x from a nonlinear measurement h(x) = sin(x),
    // linearized (via jacobianH = cos(x)) at the current estimate each step.
    const Vec1 = maryam.MatrixType(1, 1);

    const Model = struct {
        pub fn f(x: Vec1, u: Vec1) Vec1 {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: Vec1, u: Vec1) Vec1 {
            _ = x;
            _ = u;
            var m = Vec1.zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: Vec1) Vec1 {
            var out = Vec1.zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: Vec1) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };

    const EKF = extended_kalman.ExtendedKalmanFilter(1, 1, 1, Model);

    var filter = EKF{
        .x = Vec1.zero(),
        .P = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 1;
            break :blk m;
        },
        .Q = Vec1.zero(),
        .R = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 0.1;
            break :blk m;
        },
    };

    filter.predict(Vec1.zero()); // no control input this step
    var z = Vec1.zero();
    z.data[0][0] = 0.5; // measured sin(x) = 0.5
    try filter.update(z);

    try std.testing.expect(filter.x.data[0][0] > 0); // moved toward asin(0.5)
}

test "Readme.md: IteratedExtendedKalmanFilter example" {
    // Same model as the EKF example above -- IteratedExtendedKalmanFilter
    // takes the exact same Model interface, so any Model already written
    // for the EKF works here unchanged. `max_iterations` is a 5th
    // **comptime** parameter (not a struct field), since it's an
    // algorithm-shape choice, not per-instance data.
    const Vec1 = maryam.MatrixType(1, 1);

    const Model = struct {
        pub fn f(x: Vec1, u: Vec1) Vec1 {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: Vec1, u: Vec1) Vec1 {
            _ = x;
            _ = u;
            var m = Vec1.zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: Vec1) Vec1 {
            var out = Vec1.zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: Vec1) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };

    const IEKF = iterated_extended_kalman.IteratedExtendedKalmanFilter(1, 1, 1, Model, 3);

    var filter = IEKF{
        .x = Vec1.zero(),
        .P = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 1;
            break :blk m;
        },
        .Q = Vec1.zero(),
        .R = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 0.1;
            break :blk m;
        },
    };

    filter.predict(Vec1.zero()); // no control input this step
    var z = Vec1.zero();
    z.data[0][0] = 0.5; // measured sin(x) = 0.5
    try filter.update(z);

    try std.testing.expect(filter.x.data[0][0] > 0); // moved toward asin(0.5)
}

test "Readme.md: UnscentedKalmanFilter example" {
    // Same nonlinear measurement as the EKF example above (h(x) = sin(x)),
    // but Model needs no Jacobians -- UKF samples f/h directly at a small
    // set of "sigma points" instead of linearizing around the mean.
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

    const UKF = unscented_kalman.UnscentedKalmanFilter(1, 1, 1, Model);

    var filter = UKF{
        .x = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 0;
            break :blk m;
        },
        .P = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 1;
            break :blk m;
        },
        .Q = Vec1.zero(),
        .R = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 0.1;
            break :blk m;
        },
    };

    try filter.predict(Vec1.zero()); // no control input this step

    var z = Vec1.zero();
    z.data[0][0] = 0.5; // measured sin(x) = 0.5
    try filter.update(z);

    try std.testing.expect(filter.x.data[0][0] > 0); // moved toward asin(0.5)
}

test "Readme.md: SquareRootKalmanFilter example" {
    // Same model and Jacobians as the ExtendedKalmanFilter example above --
    // SquareRootKalmanFilter takes the exact same Model interface, so any
    // Model already written for the EKF works here unchanged. The only
    // difference is the field: `L` (a Cholesky factor of P, P = L @ L^T)
    // instead of `P` itself.
    const Vec1 = maryam.MatrixType(1, 1);

    const Model = struct {
        pub fn f(x: Vec1, u: Vec1) Vec1 {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: Vec1, u: Vec1) Vec1 {
            _ = x;
            _ = u;
            var m = Vec1.zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: Vec1) Vec1 {
            var out = Vec1.zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: Vec1) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };

    const SRKF = square_root_kalman.SquareRootKalmanFilter(1, 1, 1, Model);

    var filter = SRKF{
        .x = Vec1.zero(),
        .L = blk: { // P = L @ L^T = 1, same starting uncertainty as the EKF example
            var m = Vec1.zero();
            m.data[0][0] = 1;
            break :blk m;
        },
        .Q = Vec1.zero(),
        .R = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 0.1;
            break :blk m;
        },
    };

    try filter.predict(Vec1.zero()); // no control input this step
    var z = Vec1.zero();
    z.data[0][0] = 0.5; // measured sin(x) = 0.5
    try filter.update(z);

    try std.testing.expect(filter.x.data[0][0] > 0); // moved toward asin(0.5)
}

test "Readme.md: ErrorStateKalmanFilter example" {
    // Same model and Jacobians as the ExtendedKalmanFilter example above --
    // ErrorStateKalmanFilter takes the exact same Model interface, so any
    // Model already written for the EKF works here unchanged, and behaves
    // identically to it unless Model also supplies inject/resetJacobian.
    const Vec1 = maryam.MatrixType(1, 1);

    const Model = struct {
        pub fn f(x: Vec1, u: Vec1) Vec1 {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: Vec1, u: Vec1) Vec1 {
            _ = x;
            _ = u;
            var m = Vec1.zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: Vec1) Vec1 {
            var out = Vec1.zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: Vec1) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };

    const ESKF = error_state_kalman.ErrorStateKalmanFilter(1, 1, 1, Model);

    var filter = ESKF{
        .x = Vec1.zero(),
        .P = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 1;
            break :blk m;
        },
        .Q = Vec1.zero(),
        .R = blk: {
            var m = Vec1.zero();
            m.data[0][0] = 0.1;
            break :blk m;
        },
    };

    filter.predict(Vec1.zero()); // no control input this step
    var z = Vec1.zero();
    z.data[0][0] = 0.5; // measured sin(x) = 0.5
    try filter.update(z);

    try std.testing.expect(filter.x.data[0][0] > 0); // moved toward asin(0.5)
}

test "Readme.md: AdaptiveKalmanFilter example" {
    // Same h(x) = sin(x) model as the ExtendedKalmanFilter example above,
    // wrapped in AdaptiveKalmanFilter via filter_union.FilterKind's .ekf tag
    // instead of driven standalone -- proves the online-Q mechanism isn't
    // special-cased to the linear filter. Q starts badly wrong (1e-6, far
    // too small); window=5 means Q gets re-estimated from the actual
    // innovation sequence once 5 updates have run.
    const Vec1 = maryam.MatrixType(1, 1);

    const Model = struct {
        pub fn f(x: Vec1, u: Vec1) Vec1 {
            var out = x;
            out.data[0][0] += u.data[0][0];
            return out;
        }
        pub fn jacobianF(x: Vec1, u: Vec1) Vec1 {
            _ = x;
            _ = u;
            var m = Vec1.zero();
            m.data[0][0] = 1;
            return m;
        }
        pub fn h(x: Vec1) Vec1 {
            var out = Vec1.zero();
            out.data[0][0] = @sin(x.data[0][0]);
            return out;
        }
        pub fn jacobianH(x: Vec1) Vec1 {
            var m = Vec1.zero();
            m.data[0][0] = @cos(x.data[0][0]);
            return m;
        }
    };

    const Kind = filter_union.FilterKind(1, 1, 1, Model, 1);
    const AKF = adaptive_kalman.AdaptiveKalmanFilter(1, 1, 1, Kind, .{ .window = 5 });

    var filter = AKF{
        .active = .{ .ekf = extended_kalman.ExtendedKalmanFilter(1, 1, 1, Model){
            .x = Vec1.zero(),
            .P = blk: {
                var m = Vec1.zero();
                m.data[0][0] = 1;
                break :blk m;
            },
            .Q = blk: { // deliberately too small -- the point of this filter
                var m = Vec1.zero();
                m.data[0][0] = 1e-6;
                break :blk m;
            },
            .R = blk: {
                var m = Vec1.zero();
                m.data[0][0] = 0.1;
                break :blk m;
            },
        } },
    };

    const seed_q = filter.active.ekf.Q.data[0][0];

    var i: usize = 0;
    while (i < 20) : (i += 1) {
        try filter.predict(Vec1.zero());
        var z = Vec1.zero();
        z.data[0][0] = if (i % 2 == 0) 0.9 else -0.9; // sustained, large innovations
        try filter.update(z);
    }

    // After enough updates, Q has been re-estimated from the actual
    // innovation sequence instead of staying at the badly-guessed seed.
    try std.testing.expect(filter.active.ekf.Q.data[0][0] > seed_q * 100);
}
