# maryam: possible improvements for Purifier

Notes on `maryam` (https://github.com/rima1881/maryam), the linear-algebra
dependency backing `src/kalman.zig`, gathered while wiring it in, porting the
Kalman filter, and benchmarking it against real lidar/radar data. Ordered
roughly by how much they'd matter to this project.

## 1. Hardcoded `f64` — blocks the original Arduino target

`MatrixType(rows, cols)` always stores `[rows][cols]f64`
(`src/matrix.zig`). Purifier's own README says it's "optimized for the
Arduino boards, and due to resource limitations, custom algebraic functions
are implemented" — the original C++ used `float**`, not `double**`, for
exactly that reason. Most AVR-class boards do `f64` in software (slow, code
bloat) but have at least partial hardware/optimized-software `f32` support.

**Suggested change**: make the element type a parameter —
`MatrixType(comptime T: type, rows: usize, cols: usize)` (defaulting call
sites to `f64` is a 1-line change for existing users) — so an embedded target
can instantiate the same `Equation`-based filter over `f32` instead of
carrying `f64` scratch matrices it can't afford.

## 2. No solve/Cholesky — `^-1` always does full Gauss-Jordan inversion

The `Equation` DSL's only inversion path is `^-1`, which calls
`operation.invMatrix` — full Gauss-Jordan elimination with partial pivoting.
For a Kalman gain (`K = P @ H^T @ S^-1`), the standard numerical-stability
advice is "solve, don't invert": since `S = HPH^T + R` is symmetric
positive-definite by construction, solving `S @ K^T = H @ P^T` via Cholesky
is both cheaper and better-conditioned than forming `S^-1` explicitly and
multiplying.

**Suggested change**: add `operation.cholesky` / `operation.solveSPD`, and
either a new `Equation` syntax for "solve" or at least expose the primitive
so callers can drop out of the string DSL for just that one step without
hand-rolling elimination themselves.

## 3. Singularity check uses an absolute epsilon, not a relative one

In `invMatrix` (`src/operation.zig`):

```zig
if (maxCol <= std.math.floatEps(f64)) return null;
```

`floatEps(f64)` (~2.22e-16) is a fixed absolute threshold, independent of the
matrix's actual scale. A covariance matrix with small entries (e.g. tightly
converged `P`) can be relatively singular while still clearing that absolute
bar; a matrix with large entries could in principle trip it spuriously. For a
filter running continuously on live sensor data, this determines whether
`update()` returns `error.SingularMatrix` correctly.

**Suggested change**: scale the pivot threshold by the matrix's magnitude,
e.g. `maxCol <= eps * matrix_norm` (row/column max-abs is enough, no need for
a full norm), so singularity detection is consistent regardless of the
units/scale of the state being filtered.

## 4. No determinant / condition-number diagnostic

`invMatrix` computes a determinant implicitly (via the pivots) but discards
it — there's no way to ask "how close to singular is this?" short of
inverting and hoping. Kalman filter tuning commonly wants this: checking the
innovation covariance `S` for near-singularity or tracking its condition
number is a standard filter-health check (alongside NIS consistency tests).

**Suggested change**: expose `operation.determinant` (cheap, it's already
computed as a byproduct of `invMatrix`'s elimination) as its own function.


**Suggested change**: nothing to fix in maryam's code, but the README's
install section could mention this known failure mode and the tarball
workaround, since anyone hitting it will otherwise assume the package itself
is broken.

## 5. No element-wise (Hadamard) product or matrix concatenation

Lower priority for the current 1D/4-state filter, but would matter if
Purifier grows into multi-sensor fusion (stacking measurement vectors/blocks,
e.g. combining lidar + an additional sensor into one augmented `H`). Right
now building a block matrix means manually writing out `.data` by hand
outside the `Equation` DSL.

**Suggested change**: not urgent — flagging for later if the state/measurement
dimensions grow.
