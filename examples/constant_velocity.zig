//! Shared constant-velocity linear-KF pieces used by both bench drivers'
//! `runLinear()` (`imu_bench.zig`, `kitti_bench.zig`): state [px, py, vx,
//! vy], `F` built from `dt`, and the standard discretized-white-noise-
//! acceleration `Q`. Both benches compare their nonlinear filter against the
//! exact same baseline, so this was previously copy-pasted verbatim (under
//! different names) rather than shared.

const maryam = @import("maryam");

pub const StateVec = maryam.MatrixType(4, 1);
pub const StateMat = maryam.MatrixType(4, 4);
pub const ControlVec = maryam.MatrixType(1, 1);
pub const ControlMat = maryam.MatrixType(4, 1);
pub const MeasureVec = maryam.MatrixType(2, 1);
pub const MeasureMat = maryam.MatrixType(2, 4);
pub const MeasureNoise = maryam.MatrixType(2, 2);

pub fn identity() StateMat {
    var m = StateMat.zero();
    m.data[0][0] = 1;
    m.data[1][1] = 1;
    m.data[2][2] = 1;
    m.data[3][3] = 1;
    return m;
}

// Constant-velocity state transition: px += vx*dt, py += vy*dt.
pub fn stateTransition(dt: f64) StateMat {
    var m = identity();
    m.data[0][2] = dt;
    m.data[1][3] = dt;
    return m;
}

// Standard discretized-white-noise-acceleration process noise.
pub fn processNoise(dt: f64, ax: f64, ay: f64) StateMat {
    const dt2 = dt * dt;
    const dt3 = dt2 * dt;
    const dt4 = dt3 * dt;
    var m = StateMat.zero();
    m.data[0][0] = dt4 / 4 * ax;
    m.data[0][2] = dt3 / 2 * ax;
    m.data[2][0] = dt3 / 2 * ax;
    m.data[2][2] = dt2 * ax;
    m.data[1][1] = dt4 / 4 * ay;
    m.data[1][3] = dt3 / 2 * ay;
    m.data[3][1] = dt3 / 2 * ay;
    m.data[3][3] = dt2 * ay;
    return m;
}

pub fn measureH() MeasureMat {
    var m = MeasureMat.zero();
    m.data[0][0] = 1;
    m.data[1][1] = 1;
    return m;
}
