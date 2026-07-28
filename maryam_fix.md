# maryam: possible improvements for Purifier

Notes on `maryam` (https://github.com/rima1881/maryam), the linear-algebra
dependency backing `src/kalman.zig`, gathered while wiring it in, porting the
Kalman filter, and benchmarking it against real lidar/radar data.

**Update (0.2.0)**: most of what's below has since been fixed upstream — see
the "RESOLVED" items. Purifier is now pinned to 0.2.0
(`build.zig.zon`) and `kalman.zig`'s Kalman-gain equation was rewritten to
`"(S^-1 @ (H @ P))^T"` to take advantage of the new solve-fusion (item 2).

**Update (past 0.2.0, unreleased on `main`)**: `choleskyMatrix`/`sqrtMatrix`
landed (item 8) — the matrix-square-root primitive UKF/SR-KF were blocked
on. Purifier is pinned to this commit too (same version string, `0.2.0`,
since upstream hasn't bumped it for this addition yet). Both `unscented_kalman.zig`
and `square_root_kalman.zig` are now implemented on top of it — see item 9
for the primitive SR-KF's own next step is blocked on.

**Update (past that, still unreleased on `main`)**: `operation.qrMatrix`
landed (item 9) — Householder QR, plus `^Q`/`^R` equation-DSL syntax and a
new `EvalError.RankDeficient`. Purifier is pinned to this commit too. Not
yet consumed anywhere in this repo — `square_root_kalman.zig` still uses the
`P = L @ L^T` round-trip item 9 describes; switching it to a true QR-based
recursion is the natural next step now that the primitive exists.

## 1. Hardcoded `f64` — blocks the original Arduino target (open)

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

## 2. RESOLVED — solve instead of invert

0.2.0 added `operation.solveMatrix` (Gauss-Jordan on the augmented `[A | b]`
system, no explicit inverse materialized) and taught `Equation` to detect the
`... @ A^-1 @ b` pattern in a `@` chain and route through it automatically.
`A^-1` alone (nothing chained after it) still materializes the explicit
inverse, which is why Purifier's `KalmanGainK` was rewritten from
`"P @ H^T @ S^-1"` (inverse trailing, doesn't match the pattern) to
`"(S^-1 @ (H @ P))^T"` (inverse leading, does match — exploits `S`/`P` being
symmetric so `K^T = S^-1 @ H @ P`).

## 3. RESOLVED — singularity tolerance is now scaled, not absolute

`operation.zig` now has `singularityTolerance`, computing
`eps * nrows * max_abs_entry` instead of the old fixed `floatEps(f64)`
constant — consistent regardless of the matrix's scale.

## 4. RESOLVED — determinant exposed

`operation.detMatrix` is now public, and the equation DSL got a `|A|`
determinant syntax. Useful for filter-health diagnostics (condition number /
near-singularity checks on `S`) if we ever want them.

## 5. RESOLVED — Hadamard (element-wise) product

New `.` operator in the equation DSL (`A . B`), backed by
`operation.hadamardMatrix`. Not currently used anywhere in `kalman.zig`.

## 6. Still no block/concatenation support (open, low priority)

No way to build a block matrix (e.g. stacking measurement vectors from
multiple sensors into one augmented `H`) inside the `Equation` DSL — still
has to be done by hand via `.data`. Only matters if Purifier grows into
multi-sensor fusion; not needed for the current 4-state filter.

## 7. `zig fetch git+https://...` is broken (Zig-side, not maryam)

Not a maryam code issue, but worth noting since it's the first thing anyone
adding this dependency will hit: `zig fetch --save git+https://github.com/...`
fails with `error: unable to discover remote git server capabilities:
HttpConnectionClosing` — reproduced even fetching Zig's own repo tarball, so
it's a bug/quirk in Zig 0.16's `std.http.Client`, not this network or this
package. Workaround used here (three times now, for 0.1.0, 0.2.0, and this
post-0.2.0 commit): fetch the tarball with any other HTTP client, then `zig
fetch --save file:///path/to/tarball.tar.gz` (hash is computed from content,
so the resulting `build.zig.zon` entry is valid once the `url` is edited
back to the real GitHub URL).

**Suggested change**: nothing to fix in maryam's code, but the README's
install section could mention this known failure mode and the tarball
workaround, since anyone hitting it will otherwise assume the package itself
is broken.

## 8. RESOLVED — matrix square root / Cholesky exposed

`operation.choleskyMatrix(T, a)` (standard Cholesky, `a = L @ L^T`, `null` if
`a` isn't SPD) and `operation.sqrtMatrix(T, a)` (general matrix square root
via Denman-Beavers iteration, backing the equation DSL's new `^0.5` syntax)
are both public now. This was the specific gap blocking UKF (sigma points
need `sqrt((n+lambda)*P)`) and SR-KF (propagates a Cholesky factor of `P`
directly) on the Filters checklist in `Readme.md` — both are now implemented
(`unscented_kalman.zig`, `square_root_kalman.zig`).

## 9. RESOLVED — QR decomposition exposed

`operation.qrMatrix(T, a) -> ?struct { q: MatrixType(m, m), r: T }` (Householder
reflections, `a = q @ r`, `null` if `a` doesn't have full column rank) is
public now, along with `^Q`/`^R` equation-DSL syntax and a new
`EvalError.RankDeficient`. `square_root_kalman.zig`'s `update()` now uses it
directly (Potter/Carlson-form square-root update, never re-forming `P`) —
see its module doc comment for the full recursion and validation notes.

One thing this update did **not** unblock: `predict()` still round-trips
through `P = L @ L^T`, because every process-noise `Q` this repo actually
builds (`ctrv.processNoise`, `gps_ins.processNoise`) is constructed from a
low-dimensional white-noise source (`G @ Qv @ G^T`) and is therefore always
rank-deficient — neither `choleskyMatrix` nor `sqrtMatrix` can factor a
singular `Q` to get the `Q^0.5` a QR-based predict step would need. That's a
property of these specific models, not a further maryam gap; a `Model` that
exposed its process noise as a low-rank factor directly (rather than a full
`n x n` `Q`) could sidestep it, but that would be a Purifier-side API change
(the filters all take `Q: StateMat` as a runtime field today), not a maryam
one.
