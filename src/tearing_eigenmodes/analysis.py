import logging
from typing import Callable, Union, Tuple, Any, Optional, Dict, List
import numpy as np
from scipy.optimize import minimize_scalar

from .systems import TearingClassicalMHD, TearingGyrotropicMHD

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
    L_lhs: Optional[np.ndarray] = None,
    floor_eps: Optional[float] = None,
) -> float:
    """
    Extract the positive outer boundary of the connected central dominance
    interval where |T_num| >= |T_ref|.

    Uses an equation-local activity floor:
        eps_q = c_eps * eps_mach * max_{|z| <= z_max} |L_q(z)|
    ensuring scale-invariance under arbitrary complex eigenvector normalizations.

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

    if z_max is None or z_max <= 0.0:
        z_max_val = float(np.max(np.abs(z_arr)))
    else:
        z_max_val = z_max

    # 1. Define analysis-window mask first
    mask_win = np.abs(z_arr) <= z_max_val
    if not np.any(mask_win):
        return float("nan")

    eps_mach = np.finfo(float).eps

    # 2. Maximum term amplitudes within the analysis window
    max_num_win = float(np.max(num_arr[mask_win]))
    max_ref_win = float(np.max(ref_arr[mask_win]))
    max_terms_win = max(max_num_win, max_ref_win)

    if max_terms_win <= 0.0 or np.isnan(max_terms_win):
        return float("nan")

    # 3. Calculate equation-local floor
    eps_q: float
    if L_lhs is not None:
        L_arr = np.abs(np.asarray(L_lhs))
        if L_arr.size > 0 and np.any(mask_win):
            max_lhs = float(np.max(L_arr[mask_win]))
        elif L_arr.size > 0:
            max_lhs = float(np.max(L_arr))
        else:
            max_lhs = 0.0
        if max_lhs > 0.0 and np.isfinite(max_lhs):
            eps_q = float(100.0 * eps_mach * max_lhs)
        else:
            eps_q = float(100.0 * eps_mach * max_terms_win)
    else:
        eps_q = float(100.0 * eps_mach * max_terms_win)

    # Explicit floor_eps override if valid positive finite
    if floor_eps is not None:
        if np.isfinite(floor_eps) and floor_eps > 0.0:
            eps_q = floor_eps

    # 4. Numerator activity check: if numerator never reaches the floor inside the window, return nan
    if max_num_win < eps_q:
        return float("nan")

    eps_val = eps_q

    def _dominance_boundary_1d(z_side: np.ndarray, num_side: np.ndarray, ref_side: np.ndarray) -> float:
        if z_side.size < 2:
            return float("inf")

        # Find the first node where at least one term is active (>= eps_val)
        start_idx = 0
        while start_idx < z_side.size and num_side[start_idx] < eps_val and ref_side[start_idx] < eps_val:
            start_idx += 1

        if start_idx >= z_side.size:
            # All nodes below floor on this side
            return float("inf")

        # Centre node dominance check at the first active node:
        # If numerator does not dominate at the first active node, no central dominance exists
        if num_side[start_idx] < ref_side[start_idx]:
            return float("inf")

        # Walk outward from resonant center
        for i in range(start_idx, z_side.size - 1):
            if num_side[i] >= ref_side[i] and num_side[i + 1] < ref_side[i + 1]:
                # Transition bracket [i, i+1]
                if num_side[i] < eps_val and num_side[i + 1] < eps_val:
                    return float("inf")

                z_l, z_r = z_side[i], z_side[i + 1]
                q_l = np.log(num_side[i] + eps_val) - np.log(ref_side[i] + eps_val)
                q_r = np.log(num_side[i + 1] + eps_val) - np.log(ref_side[i + 1] + eps_val)

                denom = q_l - q_r
                if abs(denom) > 1e-30:
                    zc = z_l + (z_r - z_l) * (q_l / denom)
                else:
                    zc = 0.5 * (z_l + z_r)
                return float(zc)

            # If numerator continues to dominate at i+1, check if both terms are sub-floor
            if num_side[i + 1] < eps_val and ref_side[i + 1] < eps_val:
                # Connected active interval ended without a crossing
                return float("inf")

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


from dataclasses import dataclass

@dataclass
class ClassicalTermProfiles:
    z: np.ndarray
    L_b: np.ndarray
    T_F_b: np.ndarray
    T_G_b: Optional[np.ndarray]
    T_xi_b: Optional[np.ndarray]
    T_eta_b: Optional[np.ndarray]
    E_b: np.ndarray
    L_omega: np.ndarray
    T_F_omega: np.ndarray
    T_G_omega: Optional[np.ndarray]
    T_xi_omega: Optional[np.ndarray]
    T_nu_omega: Optional[np.ndarray]
    E_omega: np.ndarray


def evaluate_classical_terms(system: Any) -> ClassicalTermProfiles:
    """
    Evaluate exact equation-local signed complex term profiles for Classical MHD.
    """
    grid = system.grid
    z = grid.zg
    kx = system.kx
    a = getattr(system, "a", 1.0)
    w = getattr(system, "w", 0.0)
    shear = getattr(system, "shear", False)
    xi = getattr(system, "ξ", 0.0)
    eta = getattr(system, "η", None)
    nu = getattr(system, "ν", 0.0)
    epsilon = getattr(system, "ϵ", getattr(system, "\u03b5", getattr(system, "Hall", getattr(system, "epsilon", 0.0))))
    ky = getattr(system, "ky", 0.0)

    if (isinstance(epsilon, bool) and epsilon) or (not isinstance(epsilon, bool) and float(epsilon) > 0):
        raise NotImplementedError("Classical Hall branch (ϵ > 0) is not supported in schema version 1.")
    if not np.isclose(ky, 0.0):
        raise NotImplementedError("Classical 3D modes (ky != 0) are not supported in schema version 1.")

    sol = system.result
    u = sol["duz"]
    b = sol["dbz"]
    sigma = sol["sigma"]

    F = system.Bx
    G = getattr(system, "Ux", 0.0)

    # Grid derivatives
    u_z1 = grid.derivative(u, 1)
    u_z2 = grid.derivative(u, 2)
    u_z4 = grid.derivative(u, 4)

    b_z1 = grid.derivative(b, 1)
    b_z2 = grid.derivative(b, 2)
    b_z3 = grid.derivative(b, 3)

    Dk_b = b_z2 - kx**2 * b
    Dk_u = u_z2 - kx**2 * u

    d2F = getattr(system, "d2Bxdz", None)
    if d2F is None:
        d2F = grid.derivative(F, 2)

    flow_active = bool(w > 0 and shear)
    if flow_active:
        d2G = getattr(system, "d2Uxdz", None)
        if d2G is None:
            d2G = grid.derivative(G, 2)
    else:
        d2G = None

    # Induction equation terms
    L_b = sigma * b
    T_F_b = 1j * kx * F * u
    T_G_b = (-1j * kx * G * b) if flow_active else None

    xi_active = (abs(xi) > 0.0)
    T_xi_b = (xi * u_z1) if xi_active else None

    eta_active = (eta is not None and eta > 0.0)
    T_eta_b = (eta * Dk_b) if eta_active else None

    # Ideal envelope for induction
    ideal_b_list = [np.abs(T_F_b)]
    if flow_active and T_G_b is not None:
        ideal_b_list.append(np.abs(T_G_b))
    if xi_active and T_xi_b is not None:
        ideal_b_list.append(np.abs(T_xi_b))
    E_b = np.maximum.reduce(ideal_b_list)

    # Vorticity equation terms
    L_omega = sigma * Dk_u
    T_F_omega = 1j * kx * (F * Dk_b - d2F * b)

    if flow_active and d2G is not None:
        T_G_omega = -1j * kx * (G * Dk_u - d2G * u)
    else:
        T_G_omega = None

    T_xi_omega = (xi * (b_z3 - kx**2 * b_z1)) if xi_active else None

    nu_active = (nu is not None and nu > 0.0)
    if nu_active:
        Dk2_u = u_z4 - 2.0 * kx**2 * u_z2 + kx**4 * u
        T_nu_omega = nu * Dk2_u
    else:
        T_nu_omega = None

    # Ideal envelope for vorticity
    ideal_omega_list = [np.abs(T_F_omega)]
    if flow_active and T_G_omega is not None:
        ideal_omega_list.append(np.abs(T_G_omega))
    if xi_active and T_xi_omega is not None:
        ideal_omega_list.append(np.abs(T_xi_omega))
    E_omega = np.maximum.reduce(ideal_omega_list)

    return ClassicalTermProfiles(
        z=z,
        L_b=L_b,
        T_F_b=T_F_b,
        T_G_b=T_G_b,
        T_xi_b=T_xi_b,
        T_eta_b=T_eta_b,
        E_b=E_b,
        L_omega=L_omega,
        T_F_omega=T_F_omega,
        T_G_omega=T_G_omega,
        T_xi_omega=T_xi_omega,
        T_nu_omega=T_nu_omega,
        E_omega=E_omega,
    )


def measure_classical_scales(system: Any) -> Dict[str, float]:
    """
    Measure standardized dominance scales for Classical MHD.
    """
    profiles = evaluate_classical_terms(system)
    z = profiles.z
    a = getattr(system, "a", 1.0)
    w = getattr(system, "w", 0.0)
    z_max = w + a

    scales: Dict[str, float] = {}

    # Induction scales
    if profiles.T_eta_b is not None:
        scales["classical.bz_induction.eta_vs_ideal"] = extract_central_dominance_scale(
            z, profiles.T_eta_b, profiles.E_b, z_max=z_max, L_lhs=profiles.L_b
        )
        scales["classical.bz_induction.eta_vs_f"] = extract_central_dominance_scale(
            z, profiles.T_eta_b, profiles.T_F_b, z_max=z_max, L_lhs=profiles.L_b
        )
    else:
        scales["classical.bz_induction.eta_vs_ideal"] = float("nan")
        scales["classical.bz_induction.eta_vs_f"] = float("nan")

    if profiles.T_G_b is not None:
        scales["classical.bz_induction.g_vs_f"] = extract_central_dominance_scale(
            z, profiles.T_G_b, profiles.T_F_b, z_max=z_max, L_lhs=profiles.L_b
        )
    else:
        scales["classical.bz_induction.g_vs_f"] = float("nan")

    if profiles.T_xi_b is not None:
        scales["classical.bz_induction.xi_vs_f"] = extract_central_dominance_scale(
            z, profiles.T_xi_b, profiles.T_F_b, z_max=z_max, L_lhs=profiles.L_b
        )
    else:
        scales["classical.bz_induction.xi_vs_f"] = float("nan")

    # Vorticity scales
    if profiles.T_nu_omega is not None:
        scales["classical.uz_vorticity.nu_vs_ideal"] = extract_central_dominance_scale(
            z, profiles.T_nu_omega, profiles.E_omega, z_max=z_max, L_lhs=profiles.L_omega
        )
        scales["classical.uz_vorticity.nu_vs_f"] = extract_central_dominance_scale(
            z, profiles.T_nu_omega, profiles.T_F_omega, z_max=z_max, L_lhs=profiles.L_omega
        )
    else:
        scales["classical.uz_vorticity.nu_vs_ideal"] = float("nan")
        scales["classical.uz_vorticity.nu_vs_f"] = float("nan")

    if profiles.T_G_omega is not None:
        scales["classical.uz_vorticity.g_vs_f"] = extract_central_dominance_scale(
            z, profiles.T_G_omega, profiles.T_F_omega, z_max=z_max, L_lhs=profiles.L_omega
        )
    else:
        scales["classical.uz_vorticity.g_vs_f"] = float("nan")

    if profiles.T_xi_omega is not None:
        scales["classical.uz_vorticity.xi_vs_f"] = extract_central_dominance_scale(
            z, profiles.T_xi_omega, profiles.T_F_omega, z_max=z_max, L_lhs=profiles.L_omega
        )
    else:
        scales["classical.uz_vorticity.xi_vs_f"] = float("nan")

    return scales


@dataclass
class CGLInductionTermProfiles:
    z: np.ndarray
    L_by: np.ndarray
    T_F_by: np.ndarray
    T_eta_by: Optional[np.ndarray]
    E_by: np.ndarray
    L_bz: np.ndarray
    T_F_bz: np.ndarray
    T_eta_bz: Optional[np.ndarray]
    E_bz: np.ndarray


def evaluate_cgl_induction_terms(system: Any) -> CGLInductionTermProfiles:
    """
    Evaluate exact equation-local signed complex term profiles for non-Hall CGL induction equations.
    """
    grid = system.grid
    z = grid.zg
    kx = system.kx
    a = getattr(system, "a", 1.0)
    F = system.Bx
    H = getattr(system, "By", None)
    eta = getattr(system, "η", None)
    epsilon = getattr(system, "ϵ", getattr(system, "\u03b5", getattr(system, "Hall", getattr(system, "epsilon", 0.0))))
    ky = getattr(system, "ky", 0.0)

    if (isinstance(epsilon, bool) and epsilon) or (not isinstance(epsilon, bool) and float(epsilon) > 0):
        raise NotImplementedError("CGL Hall branch (ϵ > 0) is not supported in schema version 1.")
    if not np.isclose(ky, 0.0):
        raise NotImplementedError("CGL modes with ky != 0 are not supported in schema version 1.")

    sol = system.result
    uy = sol["duy"]
    uz = sol["duz"]
    by = sol["dby"]
    bz = sol["dbz"]
    sigma = sol["sigma"]

    # Grid derivatives
    by_z2 = grid.derivative(by, 2)
    bz_z2 = grid.derivative(bz, 2)

    Dk_by = by_z2 - kx**2 * by
    Dk_bz = bz_z2 - kx**2 * bz

    # Induction by equation: sigma*by = -kx*Bx*duy + (Bx*By/a)*duz + eta*Dk(by)
    L_by = sigma * by
    if H is not None:
        T_F_by = -kx * F * uy + (F * H / a) * uz
    else:
        T_F_by = -kx * F * uy

    eta_active = (eta is not None and eta > 0.0)
    T_eta_by = (eta * Dk_by) if eta_active else None
    E_by = np.abs(T_F_by)

    # Induction bz equation: sigma*bz = kx*Bx*duz + eta*Dk(bz)
    L_bz = sigma * bz
    T_F_bz = kx * F * uz
    T_eta_bz = (eta * Dk_bz) if eta_active else None
    E_bz = np.abs(T_F_bz)

    return CGLInductionTermProfiles(
        z=z,
        L_by=L_by,
        T_F_by=T_F_by,
        T_eta_by=T_eta_by,
        E_by=E_by,
        L_bz=L_bz,
        T_F_bz=T_F_bz,
        T_eta_bz=T_eta_bz,
        E_bz=E_bz,
    )


def measure_cgl_scales(system: Any) -> Dict[str, float]:
    """
    Measure standardized dominance scales for CGL induction equations (schema version 1).
    """
    profiles = evaluate_cgl_induction_terms(system)
    z = profiles.z
    a = getattr(system, "a", 1.0)
    z_max = a

    scales: Dict[str, float] = {}

    if profiles.T_eta_by is not None:
        scales["cgl.by_induction.eta_vs_ideal"] = extract_central_dominance_scale(
            z, profiles.T_eta_by, profiles.E_by, z_max=z_max, L_lhs=profiles.L_by
        )
        scales["cgl.by_induction.eta_vs_f"] = extract_central_dominance_scale(
            z, profiles.T_eta_by, profiles.T_F_by, z_max=z_max, L_lhs=profiles.L_by
        )
    else:
        scales["cgl.by_induction.eta_vs_ideal"] = float("nan")
        scales["cgl.by_induction.eta_vs_f"] = float("nan")

    if profiles.T_eta_bz is not None:
        scales["cgl.bz_induction.eta_vs_ideal"] = extract_central_dominance_scale(
            z, profiles.T_eta_bz, profiles.E_bz, z_max=z_max, L_lhs=profiles.L_bz
        )
        scales["cgl.bz_induction.eta_vs_f"] = extract_central_dominance_scale(
            z, profiles.T_eta_bz, profiles.T_F_bz, z_max=z_max, L_lhs=profiles.L_bz
        )
    else:
        scales["cgl.bz_induction.eta_vs_ideal"] = float("nan")
        scales["cgl.bz_induction.eta_vs_f"] = float("nan")

    return scales


def measure_eigenmode_scales(system: Any, model: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate equation terms and measure standardized physical dominance scales.

    Dispatches to model-specific term evaluators (Classical MHD vs CGL).
    Raises NotImplementedError for unrecognized equation systems.
    """
    if model is not None:
        m = model.lower()
        if "classical" in m:
            return measure_classical_scales(system)
        elif "cgl" in m or "gyrotropic" in m:
            return measure_cgl_scales(system)
        else:
            raise NotImplementedError(f"Unsupported model: {model}")

    model_hint = getattr(system, "model", None)
    if model_hint is not None:
        m = str(model_hint).lower()
        if "classical" in m:
            return measure_classical_scales(system)
        elif "cgl" in m or "gyrotropic" in m:
            return measure_cgl_scales(system)
        else:
            raise NotImplementedError(f"Unsupported system model: {model_hint}")

    model_name = system.__class__.__name__
    if "Classical" in model_name or isinstance(system, TearingClassicalMHD):
        return measure_classical_scales(system)
    elif "Gyrotropic" in model_name or "CGL" in model_name or isinstance(system, TearingGyrotropicMHD):
        return measure_cgl_scales(system)
    else:
        raise NotImplementedError(f"Unsupported system class: {model_name}. Must be Classical or Gyrotropic MHD.")


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
    Calculate the resistive inner layer thickness for tearing instability eigenmodes.

    .. deprecated:: 0.2.0
       Use :func:`measure_eigenmode_scales` and :func:`minimum_eigenmode_scale`
       for multi-scale analysis. This compatibility wrapper specifically returns
       the induction resistive scale ('classical.bz_induction.eta_vs_ideal' or
       'cgl.bz_induction.eta_vs_ideal') rather than the global multi-scale minimum.
    """
    scales = measure_eigenmode_scales(system)
    model_name = system.__class__.__name__
    if "Gyrotropic" in model_name or "CGL" in model_name or (hasattr(system, "By") and not hasattr(system, "Ux")):
        res_key = "cgl.bz_induction.eta_vs_ideal"
    else:
        res_key = "classical.bz_induction.eta_vs_ideal"

    delta_res = scales.get(res_key, float("nan"))
    if delta_res is None or np.isnan(delta_res) or np.isinf(delta_res) or delta_res <= 0:
        δ = 0.0
    else:
        δ = delta_res

    grid = system.grid
    w = getattr(system, "w", 0.0)
    a = getattr(system, "a", 1.0)

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
