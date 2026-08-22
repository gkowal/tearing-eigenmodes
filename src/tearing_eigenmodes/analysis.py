import logging
from typing import Callable, Union, Tuple, Any, Optional, Dict, List
import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

MODE_SCALE_SCHEMA_VERSION: int = 1

CLASSICAL_GRID_SCALE_KEYS: Tuple[str, ...] = (
    "classical.bz_induction.eta_vs_ideal",
    "classical.bz_induction.eta_vs_f",
    "classical.bz_induction.g_vs_f",
    "classical.bz_induction.xi_vs_f",
    "classical.uz_vorticity.nu_vs_ideal",
    "classical.uz_vorticity.nu_vs_f",
    "classical.uz_vorticity.g_vs_f",
    "classical.uz_vorticity.xi_vs_f",
)

CGL_GRID_SCALE_KEYS: Tuple[str, ...] = (
    "cgl.by_induction.eta_vs_ideal",
    "cgl.by_induction.eta_vs_f",
    "cgl.bz_induction.eta_vs_ideal",
    "cgl.bz_induction.eta_vs_f",
)


def extract_central_dominance_scale(
    z: np.ndarray,
    T_num: np.ndarray,
    T_ref: np.ndarray,
    z_max: Optional[float] = None,
    floor_eps: float = 1e-30,
) -> float:
    """
    Extract the positive outer boundary of the connected central dominance
    interval where |T_num| >= |T_ref|.

    Returns:
        - finite float > 0: positive outer boundary of central connected dominance.
        - np.inf: term is active but has no finite central crossing in |z| <= z_max
                  (e.g., dominates the whole interval, or never dominates at center).
        - np.nan: term is inactive, invalid, empty, or all inputs are below activity floor.
    """
    z_arr = np.asarray(z, dtype=float)
    num_arr = np.abs(np.asarray(T_num))
    ref_arr = np.abs(np.asarray(T_ref))

    if z_arr.size == 0 or num_arr.size == 0 or ref_arr.size == 0:
        return float("nan")

    if not (np.all(np.isfinite(z_arr)) and np.all(np.isfinite(num_arr)) and np.all(np.isfinite(ref_arr))):
        return float("nan")

    # Activity floor check: if both profiles are entirely negligible
    max_activity = max(float(np.max(num_arr)), float(np.max(ref_arr)))
    if max_activity < 1e-15:
        return float("nan")

    if z_max is None or z_max <= 0.0:
        z_max_val = float(np.max(np.abs(z_arr)))
    else:
        z_max_val = z_max

    def _dominance_boundary_1d(z_side: np.ndarray, num_side: np.ndarray, ref_side: np.ndarray) -> float:
        if z_side.size < 2:
            return float("inf")

        # Centre node check:
        # If numerator does not dominate at the resonant center, no central dominance region exists
        if num_side[0] < ref_side[0]:
            return float("inf")

        # Walk outward from resonant center
        for i in range(z_side.size - 1):
            if num_side[i] >= ref_side[i] and num_side[i + 1] < ref_side[i + 1]:
                z_l, z_r = z_side[i], z_side[i + 1]
                q_l = np.log(num_side[i] + floor_eps) - np.log(ref_side[i] + floor_eps)
                q_r = np.log(num_side[i + 1] + floor_eps) - np.log(ref_side[i + 1] + floor_eps)

                denom = q_l - q_r
                if abs(denom) > 1e-30:
                    zc = z_l + (z_r - z_l) * (q_l / denom)
                else:
                    zc = 0.5 * (z_l + z_r)
                return float(zc)

        # Reached outer boundary of the analysis window while remaining dominant
        return float("inf")

    # Positive side (z >= 0, within z_max)
    pos_mask = (z_arr >= 0.0)
    pos_indices = np.where(pos_mask)[0]
    if pos_indices.size > 0:
        sort_order = np.argsort(z_arr[pos_indices])
        sorted_pos_idx = pos_indices[sort_order]

        z_pos = z_arr[sorted_pos_idx]
        num_pos = num_arr[sorted_pos_idx]
        ref_pos = ref_arr[sorted_pos_idx]

        in_window = np.where(z_pos <= z_max_val)[0]
        if in_window.size > 0:
            last_idx = min(z_pos.size, in_window[-1] + 2)
            z_pos_win = z_pos[:last_idx]
            num_pos_win = num_pos[:last_idx]
            ref_pos_win = ref_pos[:last_idx]
            scale_pos = _dominance_boundary_1d(z_pos_win, num_pos_win, ref_pos_win)
            if np.isfinite(scale_pos) and scale_pos > z_max_val:
                scale_pos = float("inf")
        else:
            scale_pos = float("inf")
    else:
        scale_pos = float("nan")

    # Negative side (z <= 0, within z_max)
    neg_mask = (z_arr <= 0.0)
    neg_indices = np.where(neg_mask)[0]
    if neg_indices.size > 0:
        dist_neg = np.abs(z_arr[neg_indices])
        sort_order = np.argsort(dist_neg)
        sorted_neg_idx = neg_indices[sort_order]

        z_neg = dist_neg[sort_order]
        num_neg = num_arr[sorted_neg_idx]
        ref_neg = ref_arr[sorted_neg_idx]

        in_window = np.where(z_neg <= z_max_val)[0]
        if in_window.size > 0:
            last_idx = min(z_neg.size, in_window[-1] + 2)
            z_neg_win = z_neg[:last_idx]
            num_neg_win = num_neg[:last_idx]
            ref_neg_win = ref_neg[:last_idx]
            scale_neg = _dominance_boundary_1d(z_neg_win, num_neg_win, ref_neg_win)
            if np.isfinite(scale_neg) and scale_neg > z_max_val:
                scale_neg = float("inf")
        else:
            scale_neg = float("inf")
    else:
        scale_neg = float("nan")

    # Combine sides
    finite_pos = np.isfinite(scale_pos) and scale_pos > 0.0
    finite_neg = np.isfinite(scale_neg) and scale_neg > 0.0

    if finite_pos and finite_neg:
        max_s = max(scale_pos, scale_neg)
        min_s = min(scale_pos, scale_neg)
        if max_s > 0 and (max_s - min_s) / max_s > 0.2:
            logger.debug(f"Asymmetric dominance crossing: pos={scale_pos:.4e}, neg={scale_neg:.4e}")
        return min_s
    elif finite_pos:
        return scale_pos
    elif finite_neg:
        return scale_neg
    elif np.isinf(scale_pos) or np.isinf(scale_neg):
        return float("inf")
    else:
        return float("nan")


def minimum_eigenmode_scale(
    scales: Dict[str, float],
    model: str = "classical",
) -> Tuple[Optional[float], Optional[str]]:
    """
    Select the minimum valid physical scale and its standardized key.

    Filters candidates against the model's key whitelist and retains only
    finite values > 0.0. Deterministic key order resolves exact ties.
    Returns (None, None) if no valid candidate exists.
    """
    if model.lower().startswith("cgl") or model.lower() == "gyrotropic":
        candidate_keys = CGL_GRID_SCALE_KEYS
    else:
        candidate_keys = CLASSICAL_GRID_SCALE_KEYS

    min_scale: Optional[float] = None
    min_key: Optional[str] = None

    for key in candidate_keys:
        val = scales.get(key)
        if val is not None and np.isfinite(val) and val > 0.0:
            if min_scale is None or val < min_scale:
                min_scale = val
                min_key = key

    return min_scale, min_key


def make_fast_interpolator(grid: Any) -> Callable[[float, np.ndarray], float]:
    """
    Robust single-point interpolator for fields defined on grid.zg,
    using global barycentric interpolation on the mapped Chebyshev–Gauss nodes.
    """
    # Cache mapped Chebyshev–Gauss nodes (ascending, matching grid.zg ordering)
    xj = np.clip(grid.zg / np.sqrt(grid.C**2 + grid.zg**2), -1.0, 1.0)
    N = xj.size

    # Barycentric weights for Chebyshev–Gauss nodes:
    # w_j ∝ (-1)^(j-1) * sin(theta_j), theta_j=(2j-1)π/(2N)
    j = np.arange(1, N + 1)
    theta = (2*j - 1) * np.pi / (2*N)
    wj = ((-1.0)**(j - 1)) * np.sin(theta)

    # theta construction corresponds to x=cos(theta) in descending order; reverse to match xj ascending
    wj = wj[::-1]

    eps = 50 * np.finfo(float).eps

    def fast_interpolate(z_target: float, f_values: np.ndarray) -> float:
        f_values = np.asarray(f_values)
        x = z_target / np.sqrt(grid.C**2 + z_target**2)

        # Tolerant "hit a node" shortcut
        idx = int(np.argmin(np.abs(x - xj)))
        if np.abs(x - xj[idx]) <= eps:
            return float(f_values[idx])

        tmp = wj / (x - xj)
        return float(np.dot(tmp, f_values) / np.sum(tmp))

    return fast_interpolate


def inner_layer_thickness(system: Any, δtol: float = 1e-3, maxiter: int = 20) -> Tuple[float, int, int]:
    """
    Calculate the inner layer thickness δ for the tearing instability eigenmode.
    """
    grid = system.grid
    sol  = system.result

    C = grid.C
    s = getattr(system, 'shear', False)
    a = system.a
    w = getattr(system, 'w', 0.0)
    S = system.S
    ξ = getattr(system, 'ξ', 0.0)
    α = system.kx * a
    u = sol['duz']
    b = sol['dbz']
    F = system.Bx
    G = getattr(system, 'Ux', 0.0)

    if s:
        Ti = np.abs(1j * α * (u * F - G * b) + ξ * grid.derivative(u, 1))
    else:
        Ti = np.abs(1j * α *  u * F          + ξ * grid.derivative(u, 1))
    Tn = np.abs(grid.derivative(b, 2) - α**2 * b) / S
    Td = Tn - Ti

    I   = np.where((0.0 <= grid.zg) & (grid.zg <= (w + 2 * a)))
    zin = grid.zg[I]
    Tin = Tn[I] - Ti[I]

    # Build interpolator once
    fast_interp = make_fast_interpolator(grid)

    δ = 0.0
    # check if Td changes sign in the interval
    if Tin.min() <= 0.0 and Tin.max() >= 0.0:
        # start from z at which Tin is maximum and search at which distance Td becomes negative
        idx = Tin.argmax()
        while Tin[idx] > 0.0 and idx < (Tin.size - 1):
            idx += 1
        idx -= 1
        if idx < (Tin.size - 1):
            # endpoints of the bracket on the sliced arrays
            zl, zh = float(zin[idx]), float(zin[idx + 1])
            tl, th = float(Tin[idx]), float(Tin[idx + 1])

            # If bracket does not change sign, fall back to choosing the best of endpoints/midpoint
            if tl * th > 0.0:
                zm = 0.5 * (zl + zh)
                tm = fast_interp(zm, Td)
                # pick the point with smallest absolute value
                best = min(((abs(tl), zl), (abs(tm), zm), (abs(th), zh)), key=lambda x: x[0])[1]
                δ = best
            else:
                it = 0
                while 2.0 * (zh - zl) > δtol * (zh + zl) and it < maxiter:
                    zm = 0.5 * (zl + zh)
                    tm = fast_interp(zm, Td)
                    if tm == 0.0:
                        zl = zh = zm
                        break
                    if tl * tm > 0:
                        zl, tl = zm, tm
                    else:
                        zh, th = zm, tm
                    it += 1
                δ = 0.5 * (zl + zh)

    I = np.where(np.abs(grid.zg) <= δ)
    nin = max(1, I[0].size)

    I = np.where(np.abs(grid.zg) <= (w + a))
    nwa = I[0].size

    return δ, nin, nwa


def find_peak_location(u0: np.ndarray, b0: np.ndarray, grid: Any, a: float = 1.0, w: float = 0.0, ztol: float = 1.0e-3, maxiter: int = 50) -> Tuple[float, int]:
    """
    Find the closest positive peak location among u, b, and their first two derivatives.
    """
    u1 = grid.derivative(u0, 1)
    u2 = grid.derivative(u0, 2)
    b1 = grid.derivative(b0, 1)
    b2 = grid.derivative(b0, 2)

    fields = {'u0': u0, 'u1': u1, 'u2': u2, 'b0': b0, 'b1': b1, 'b2': b2}

    z_peak = w + 2 * a
    field_idx = None
    peak_idx = None

    # Quickly find the peak on the base grid before refining the winning field.
    I = np.where((0.0 <= grid.zg) & (grid.zg <= (w + 2 * a)))[0]
    z = grid.zg[I]
    for f, values in fields.items():
        i = np.abs(values[I]).argmax()
        idx = I[i]

        if 0.0 < z[i] < z_peak:
            z_peak = z[i]
            field_idx = f
            peak_idx = idx

    # Refine using one neighboring grid point on each side of the coarse peak.
    if field_idx is not None and peak_idx is not None and 0 < peak_idx < (grid.zg.size - 1):
        zl = max(0.0, float(grid.zg[peak_idx - 1]))
        zh = min(a, float(grid.zg[peak_idx + 1]))

        if zl < z_peak < zh:
            def objective(zz):
                return -float(np.abs(grid.interpolate(zz, fields[field_idx])))

            result = minimize_scalar(
                objective,
                bounds=(zl, zh),
                method='bounded',
                options={'xatol': ztol * z_peak, 'maxiter': maxiter},
            )

            candidates = [float(z_peak), zl, zh]
            if result.success and zl <= result.x <= zh:
                candidates.append(float(result.x))

            z_peak = max(candidates, key=lambda zz: -objective(zz))

    I_outer = np.where(np.abs(grid.zg) <= (w + a))
    n = I_outer[0].size
    if w > 0:
        I_inner = np.where(np.abs(grid.zg) <= w)
        n -= I_inner[0].size

    return z_peak, n
