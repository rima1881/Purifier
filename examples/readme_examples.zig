//! The exact code shown in Readme.md's "Using this as a library" section,
//! kept here as tests so the docs can't silently drift out of sync with the
//! actual API.

const std = @import("std");
const Purifier = @import("Purifier");
const kalman = Purifier.kalman;
const extended_kalman = Purifier.extended_kalman;
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
