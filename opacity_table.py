import os
import numpy as np


_OPACITY_CACHE = {}


def table_filename(X, Z):
    X_name = f"{X:.2f}".replace(".", "p")
    Z_name = f"{Z:.2f}".replace(".", "p")
    return f"op_X{X_name}_Z{Z_name}_logT_logR.txt"

def find_table(X, Z):
    here = os.path.dirname(__file__)
    fname = table_filename(X, Z)

    paths = [
        os.path.join(here, "opacity_tables", fname),
        os.path.join(here, fname),
    ]

    for path in paths:
        if os.path.exists(path):
            return path

    return None

def opacity_source(X=0.70, Y=0.28, Z=0.02):
    table_path = find_table(X, Z)

    if table_path is not None:
        return f"OP table: {os.path.basename(table_path)}"

    return "Using Calculated Opacity"

def fill_invalid_values(log_kappa):
    out = log_kappa.copy()

    for i in range(out.shape[0]):
        valid = np.where(np.isfinite(out[i]))[0]

        if len(valid) == 0:
            continue

        invalid = np.where(~np.isfinite(out[i]))[0]

        for j in invalid:
            nearest = valid[np.argmin(np.abs(valid - j))]
            out[i, j] = out[i, nearest]

    for j in range(out.shape[1]):
        valid = np.where(np.isfinite(out[:, j]))[0]

        if len(valid) == 0:
            continue

        invalid = np.where(~np.isfinite(out[:, j]))[0]

        for i in invalid:
            nearest = valid[np.argmin(np.abs(valid - i))]
            out[i, j] = out[nearest, j]

    if np.any(~np.isfinite(out)):
        raise ValueError("Opacity table contains invalid values")

    return out


def read_op_table(table_path):
    rows = []

    with open(table_path, "r") as f:
        for line in f:
            parts = line.split()

            if len(parts) < 6:
                continue

            try:
                logT = float(parts[0])
                logR = float(parts[1])
                logkappa = float(parts[3])
            except ValueError:
                continue

            rows.append((logT, logR, logkappa))

    if len(rows) == 0:
        raise ValueError(f"No opacity data found in {table_path}")

    data = np.array(rows)

    logT_grid = np.unique(data[:, 0])
    logR_grid = np.unique(data[:, 1])

    logT_grid.sort()
    logR_grid.sort()

    log_kappa = np.full((len(logT_grid), len(logR_grid)), np.nan)

    T_index = {v: i for i, v in enumerate(logT_grid)}
    R_index = {v: j for j, v in enumerate(logR_grid)}

    for logT, logR, logk in data:
        i = T_index[logT]
        j = R_index[logR]

        if np.isfinite(logk) and logk < 9.0:
            log_kappa[i, j] = logk

    log_kappa = fill_invalid_values(log_kappa)

    return logT_grid, logR_grid, log_kappa


def kappa_electron(rho, T, X):
    # electron scattering
    return 0.2 * (1.0 + X)


def kappa_kramers_bf(rho, T, X, Z):
    # bound-free opacity
    return 4e25 * Z * (1.0 + X) * rho * T**(-3.5)


def kappa_kramers_ff(rho, T, X, Y, Z):
    # free-free opacity
    return 4e22 * (X + Y) * (1.0 + X) * rho * T**(-3.5)


def kappa_Hminus(rho, T, Z):
    # H-minus opacity estimate
    base = 2.5e-31 * (Z / 0.02) * np.sqrt(np.maximum(rho, 1.0e-300)) * T**9

    return base


def analytic_opacity(rho, T, X, Y, Z):
    k_rad = (
        kappa_electron(rho, T, X)
        + kappa_kramers_bf(rho, T, X, Z)
        + kappa_kramers_ff(rho, T, X, Y, Z)
        + kappa_Hminus(rho, T, Z)
    )

    return np.maximum(k_rad, 1.0e-30)


def build_analytic_table(X, Y, Z):
    logT_grid = np.linspace(3.5, 7.5, 170)
    logR_grid = np.linspace(-8.0, 1.0, 91)

    LT, LR = np.meshgrid(logT_grid, logR_grid, indexing="ij")

    T = 10.0 ** LT
    T6 = T / 1.0e6
    rho = 10.0 ** LR * T6 ** 3

    kappa = analytic_opacity(rho, T, X, Y, Z)
    log_kappa = np.log10(np.maximum(kappa, 1e-10))

    return logT_grid, logR_grid, log_kappa


def opacity_grid(X=0.70, Y=0.28, Z=0.02):
    table_path = find_table(X, Z)

    if table_path is not None:
        key = ("OP", os.path.abspath(table_path))
    else:
        key = ("analytic", round(X, 8), round(Y, 8), round(Z, 8))

    if key in _OPACITY_CACHE:
        return _OPACITY_CACHE[key]

    if table_path is not None:
        grid = read_op_table(table_path)
    else:
        grid = build_analytic_table(X, Y, Z)

    _OPACITY_CACHE[key] = grid

    return grid


def interpolate_opacity(rho, T, logT_grid, logR_grid, log_kappa):
    rho = np.asarray(rho, dtype=float)
    T = np.asarray(T, dtype=float)

    scalar = rho.ndim == 0 and T.ndim == 0

    T6 = T / 1.0e6
    R = rho / np.maximum(T6 ** 3, 1e-300)

    logT = np.log10(np.maximum(T, 1.0))
    logR = np.log10(np.maximum(R, 1e-300))

    logT = np.clip(logT, logT_grid[0], logT_grid[-1] - 1e-12)
    logR = np.clip(logR, logR_grid[0], logR_grid[-1] - 1e-12)

    dT = logT_grid[1] - logT_grid[0]
    dR = logR_grid[1] - logR_grid[0]

    iT = ((logT - logT_grid[0]) / dT).astype(int)
    iR = ((logR - logR_grid[0]) / dR).astype(int)

    iT = np.clip(iT, 0, len(logT_grid) - 2)
    iR = np.clip(iR, 0, len(logR_grid) - 2)

    fT = (logT - logT_grid[iT]) / dT
    fR = (logR - logR_grid[iR]) / dR

    k00 = log_kappa[iT, iR]
    k10 = log_kappa[iT + 1, iR]
    k01 = log_kappa[iT, iR + 1]
    k11 = log_kappa[iT + 1, iR + 1]

    logk = (
        (1.0 - fT) * (1.0 - fR) * k00
        + fT * (1.0 - fR) * k10
        + (1.0 - fT) * fR * k01
        + fT * fR * k11
    )

    kappa = 10.0 ** logk

    return float(kappa) if scalar else kappa


def opacity(rho, T, X=0.70, Y=0.28, Z=0.02):
    logT_grid, logR_grid, log_kappa = opacity_grid(X, Y, Z)
    return interpolate_opacity(rho, T, logT_grid, logR_grid, log_kappa)
