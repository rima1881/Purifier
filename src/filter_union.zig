//! A tagged union over every non-adaptive filter variant in this package
//! (`kalman.KalmanFilter`, `extended_kalman.ExtendedKalmanFilter`,
//! `iterated_extended_kalman.IteratedExtendedKalmanFilter`,
//! `unscented_kalman.UnscentedKalmanFilter`,
//! `square_root_kalman.SquareRootKalmanFilter`,
//! `error_state_kalman.ErrorStateKalmanFilter`), all monomorphized for the
//! same `(n, k, m, Model)` (plus IEKF's own `iekf_iterations`, needed to name
//! its concrete type even when IEKF isn't the active variant at runtime).
//!
//! The one and only reason this exists: it lets
//! `adaptive_kalman.AdaptiveKalmanFilter` hold *any one* of these six
//! algorithms behind a single field and drive it generically -- calling
//! `predict()`/`update()` through the union, then reading back whichever
//! variant's own `last_K`/`last_y` (see `kalman.zig`'s doc comment on those
//! fields) to re-estimate `Q` -- without caring which specific algorithm is
//! doing the underlying math, or duplicating six copies of the same
//! online-`Q` bookkeeping.
//!
//! `linear`'s `KalmanFilter(n, k, m)` doesn't reference `Model` at all (it
//! takes `F`/`H` as plain fields instead of a model namespace), but `Model`
//! still has to be a genuine, full EKF-compatible namespace (`f`/
//! `jacobianF`/`h`/`jacobianH`) even for callers who only ever intend to use
//! the `linear` tag: `AdaptiveKalmanFilter.predict()`/`update()` (see
//! `adaptive_kalman.zig`) switch on `active`'s tag, which is a *runtime*
//! value, so every switch arm -- including the five nonlinear ones -- has to
//! type-check and compile, not just the arm actually reached at runtime. A
//! placeholder `Model` with no decls at all will fail to compile as soon as
//! any `AdaptiveKalmanFilter` built from it calls `predict()`/`update()`,
//! even if `active` is always tagged `.linear` in practice.

const kalman = @import("kalman.zig");
const extended_kalman = @import("extended_kalman.zig");
const iterated_extended_kalman = @import("iterated_extended_kalman.zig");
const unscented_kalman = @import("unscented_kalman.zig");
const square_root_kalman = @import("square_root_kalman.zig");
const error_state_kalman = @import("error_state_kalman.zig");

pub fn FilterKind(comptime n: usize, comptime k: usize, comptime m: usize, comptime Model: type, comptime iekf_iterations: usize) type {
    return union(enum) {
        linear: kalman.KalmanFilter(n, k, m),
        ekf: extended_kalman.ExtendedKalmanFilter(n, k, m, Model),
        iekf: iterated_extended_kalman.IteratedExtendedKalmanFilter(n, k, m, Model, iekf_iterations),
        ukf: unscented_kalman.UnscentedKalmanFilter(n, k, m, Model),
        srkf: square_root_kalman.SquareRootKalmanFilter(n, k, m, Model),
        eskf: error_state_kalman.ErrorStateKalmanFilter(n, k, m, Model),
    };
}
