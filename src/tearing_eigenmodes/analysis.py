import numpy as np
from scipy.optimize import minimize_scalar

def make_fast_interpolator(grid):
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

    def fast_interpolate(z_target, f_values):
        f_values = np.asarray(f_values)
        x = z_target / np.sqrt(grid.C**2 + z_target**2)

        # Tolerant "hit a node" shortcut
        idx = int(np.argmin(np.abs(x - xj)))
        if np.abs(x - xj[idx]) <= eps:
            return float(f_values[idx])

        tmp = wj / (x - xj)
        return float(np.dot(tmp, f_values) / np.sum(tmp))

    return fast_interpolate


def inner_layer_thickness(system, δtol=1e-3, maxiter=20):
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
                δ = float(best)
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
                δ = float(0.5 * (zl + zh))
    else:
        # no region where Td >= 0: set δ to zero (no inner layer detected)
        δ = 0.0

    I = np.where(np.abs(grid.zg) <= δ)
    nin = max(1, I[0].size)

    I = np.where(np.abs(grid.zg) <= (w + a))
    nwa = I[0].size

    return δ, nin, nwa


def find_peak_location(u0, b0, grid, a=1.0, w=0.0, ztol=1.0e-3, maxiter=50):
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
    if field_idx is not None and 0 < peak_idx < (grid.zg.size - 1):
        zl = max(0.0, float(grid.zg[peak_idx - 1]))
        zh = min(float(a), float(grid.zg[peak_idx + 1]))

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

    I = np.where(np.abs(grid.zg) <= (w + a))
    n = I[0].size
    if w > 0:
        I = np.where(np.abs(grid.zg) <= w)
        n -= I[0].size

    return z_peak, n
