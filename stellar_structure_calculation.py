from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import root
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
from math import radians, sin, cos
import matplotlib.ticker as tck
from matplotlib.ticker import NullFormatter
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as mticker
# %matplotlib widget

from opacity_table import opacity, opacity_source, opacity_grid

c_light  = 2.99792458e10
h        = 6.6260755e-27
G        = 6.67259e-8
a_rad    = 7.5646e-15
sigma = 5.67051e-5
k_B      = 1.380658e-16
N_A      = 6.022142e23
m_H = 1.6735575e-24

mp = 1.6726231e-24
me = 9.1093897e-28
qe = 4.8032068e-10
mn = 1.6749286e-24
am = 1.66053886e-24
a0 = 5.29177e-9
eV = 1.6021772e-12
Rg = 8.314
au = 1.496e13
pc = 206265 * au

M_sun = 1.989e33
R_sun = 6.96e10
L_sun = 3.826e33

Re = 6.378e8
Me = 5.9742e27
Mj = 317.83 * Me
Rj = 11.209 * Re
yr = 365 * 24 * 3600
st = 6.65e-25

def mean_molecular_weights(X, Y, Z):
    inv_mu_I = X + Y / 4.0 + Z / 14.0                  # fully ionized
    inv_mu_e = X + Y / 2.0 + Z / 2.0                   
    inv_mu = inv_mu_I + inv_mu_e                       # SI eq 1.50
    return 1.0 / inv_mu, 1.0 / inv_mu_I, 1.0 / inv_mu_e


def density(P, T, mu):
    P_rad = a_rad * T**4 / 3.0                          # SI Eq. 4.32
    P_gas = P - P_rad

    if np.isscalar(P_gas):
        if P_gas <= 0:
            P_gas = 1.0e-300 * abs(P)                    # avoid P<0
    else:
        P_gas = np.where(P_gas > 0, P_gas, 1.0e-10 * np.abs(P)) 

    return P_gas * mu * m_H / (k_B * T) 


def adiabatic_gradient(P, T):
    P_rad = a_rad * T**4 / 3.0
    beta = np.clip((P - P_rad) / P, 1.0e-300, 1.0)        
    return (8.0 - 6.0 * beta) / (32.0 - 24.0 * beta - 3.0 * beta**2)  # SSE eq 13.12


def weak_screening_factor(rho, T, X, Y, Z, mu, Z1=1, Z2=1):
    zeta_sum = 2.0 * X + 1.5 * Y + 4.0 * Z              # approximation from SSE eq 18.47
    gamma = mu * zeta_sum                               # SSE Eq. 18.47
    T7 = np.maximum(T / 1.0e7, 1.0e-300)

    ekt = 5.92e-3 * Z1 * Z2 * np.sqrt(np.maximum(rho * gamma, 0.0)) / T7**1.5  # SSE eq 18.57

    return np.exp(ekt)                                  # SSE Eq. 18.56


def epsilon_pp(rho, T, X, Y, Z, mu):
    T9 = np.maximum(T / 1.0e9, 1.0e-300)
    T913 = T9**(1.0 / 3.0)

    g11 = 1.0 + 3.82 * T9 + 1.51 * T9**2 + 0.144 * T9**3 - 0.0114 * T9**4  # SSE eq 18.63
    psi_pp = 1.0                                        # approximation
    f11 = weak_screening_factor(rho, T, X, Y, Z, mu, Z1=1, Z2=1)

    e = -3.381 / T913                                 # SSE Eq. 18.63

    return (
        2.57e4
        * psi_pp
        * f11
        * g11
        * rho
        * X**2
        * T9**(-2.0 / 3.0)
        * np.exp(e)
    )                                                    # SSE Eq. 18.63


def epsilon_CNO(rho, T, X, Y, Z, X_CNO=None):
    T9 = np.maximum(T / 1.0e9, 1.0e-300)
    T913 = T9**(1.0 / 3.0)

    if X_CNO is None:
        X_CNO = Z                                       # approximation

    g14 = 1.0 - 2.00 * T9 + 3.41 * T9**2 - 2.43 * T9**3  # SSE Eq. 18.65
    g14 = np.maximum(g14, 0.0)

    e = -15.231 / T913 - (T9 / 0.8)**2                 # SSE Eq. 18.65
    e = np.clip(e, -1000.0, 0.0)

    return (
        8.24e25
        * g14
        * rho
        * X
        * X_CNO
        * T9**(-2.0 / 3.0)
        * np.exp(e)
    )                                                    # SSE Eq. 18.65


def epsilon_total(rho, T, X, Y, Z, mu):
    return epsilon_pp(rho, T, X, Y, Z, mu) + epsilon_CNO(rho, T, X, Y, Z)


def get_opacity(rho, T, X, Y, Z):
    return opacity(rho, T, X=X, Y=Y, Z=Z)


def derivs_m(m, y, X, Y, Z, mu):
    l, P, r, T = y

    P = max(float(P), 1.0e-300)
    T = max(float(T), 1.0e-300)
    r = max(float(r), 1.0e-300)
    m = max(float(m), 1.0e-300)

    rho = max(float(density(P, T, mu)), 1.0e-300)        # avoid zero
    eps = epsilon_total(rho, T, X, Y, Z, mu)        
    kap = get_opacity(rho, T, X, Y, Z)

    nabla_rad = (
        3.0 * kap * l * P
        / (16.0 * np.pi * a_rad * c_light * G * m * T**4)
    )                                                    # SSE eq 10.6

    nabla_ad = adiabatic_gradient(P, T)
    nabla = min(nabla_rad, nabla_ad)              

    dl_dm = eps                                          
    dP_dm = -G * m / (4.0 * np.pi * r**4)
    dr_dm = 1.0 / (4.0 * np.pi * r**2 * rho)
    dT_dm = -G * m * T * nabla / (4.0 * np.pi * r**4 * P)

    return np.array([dl_dm, dP_dm, dr_dm, dT_dm])


def derivs_q(q, y, M_total, X, Y, Z, mu):
    m = q * M_total
    return M_total * derivs_m(m, y, X, Y, Z, mu)


def load1(m1, Pc, Tc, X, Y, Z, mu):
    rho_c = density(Pc, Tc, mu)
    eps_c = epsilon_total(rho_c, Tc, X, Y, Z, mu)
    kap_c = get_opacity(rho_c, Tc, X, Y, Z)

    r1 = (3.0 * m1 / (4.0 * np.pi * rho_c))**(1.0 / 3.0)  # SSE Eq. 11.3
    l1 = eps_c * m1                                      # SSE Eq. 11.4

    P1 = (
        Pc
        - (3.0 * G / (8.0 * np.pi))
        * (4.0 * np.pi / 3.0)**(4.0 / 3.0)
        * rho_c**(4.0 / 3.0)
        * m1**(2.0 / 3.0)
    )                                                     # SSE Eq. 11.6

    nabla_ad_c = adiabatic_gradient(Pc, Tc)

    nabla_rad_c = (
        3.0 * kap_c * eps_c * Pc
        / (16.0 * np.pi * a_rad * c_light * G * Tc**4)
    )                                                     # SSE Eq. 10.6 at center

    if nabla_rad_c > nabla_ad_c:
        lnT1 = (
            np.log(Tc)
            - (np.pi / 6.0)**(1.0 / 3.0)
            * G
            * nabla_ad_c
            * rho_c**(4.0 / 3.0)
            / Pc
            * m1**(2.0 / 3.0)
        )
        T1 = np.exp(lnT1)
    else:
        T1 = (
            Tc**4
            - (kap_c * eps_c / (2.0 * a_rad * c_light))
            * (3.0 / (4.0 * np.pi))**(2.0 / 3.0)
            * rho_c**(4.0 / 3.0)
            * m1**(2.0 / 3.0)
        )**0.25                                          # SSE Eq. 11.9

    return np.array([l1, P1, r1, T1])


def load2(M_total, R_star, L_star, X, Y, Z, mu):
    g_surf = G * M_total / R_star**2
    T_surf = (L_star / (4.0 * np.pi * R_star**2 * sigma))**0.25

    kap_surf = 0.4                                       # initial guess of opacity

    for _ in range(30):
        P_surf = (2.0 / 3.0) * g_surf / kap_surf     
        rho_surf = density(P_surf, T_surf, mu)
        kap_new = get_opacity(rho_surf, T_surf, X, Y, Z)

        if abs(kap_new - kap_surf) / max(kap_new, 1.0e-30) < 1.0e-6:
            break

        kap_surf = 0.5 * (kap_new + kap_surf) 
        
    return np.array([L_star, P_surf, R_star, T_surf])


def integrate_outward(m1, m_fit, y1, M_total, X, Y, Z, mu):
    q1 = m1 / M_total
    q_fit = m_fit / M_total

    return solve_ivp(
        derivs_q,
        (q1, q_fit),
        y1,
        args=(M_total, X, Y, Z, mu),
        method="LSODA",
        rtol=1.0e-8,
        atol=1.0e-10,
        max_step=(q_fit - q1) * 0.01,
    )


def integrate_inward(M_total, m_fit, y_surface, X, Y, Z, mu):
    q_fit = m_fit / M_total

    return solve_ivp(
        derivs_q,
        (1.0, q_fit),
        y_surface,
        args=(M_total, X, Y, Z, mu),
        method="LSODA",
        rtol=1.0e-8,
        atol=1.0e-10,
        max_step=(1.0 - q_fit) * 1.0e-3,
    )


def shootf_residual(params, M_total, m_fit, m_inner, X, Y, Z, mu, scales):
    Pc = params[0] * scales["Pc"]
    Tc = params[1] * scales["Tc"]
    R_star = params[2] * scales["R"]
    L_star = params[3] * scales["L"]

    if Pc <= 0 or Tc <= 0 or R_star <= 0 or L_star <= 0:
        return np.full(4, 1.0e3)

    try:
        y_inner = load1(m_inner, Pc, Tc, X, Y, Z, mu)
        y_surface = load2(M_total, R_star, L_star, X, Y, Z, mu)

        if np.any(~np.isfinite(y_inner)) or np.any(~np.isfinite(y_surface)):
            return np.full(4, 1.0e3)

        sol_out = integrate_outward(m_inner, m_fit, y_inner, M_total, X, Y, Z, mu)
        sol_in = integrate_inward(M_total, m_fit, y_surface, X, Y, Z, mu)

        if not (sol_out.success and sol_in.success):
            return np.full(4, 1.0e3)

        y_out = sol_out.y[:, -1]
        y_in = sol_in.y[:, -1]

        if np.any(~np.isfinite(y_out)) or np.any(~np.isfinite(y_in)):
            return np.full(4, 1.0e3)

        return np.log(np.maximum(y_out, 1.0e-300)) - np.log(np.maximum(y_in, 1.0e-300))

    except Exception:
        return np.full(4, 1.0e3)


def final_profile(M_total, m_fit, m_inner, Pc, Tc, R_star, L_star, X, Y, Z, mu):
    y_inner = load1(m_inner, Pc, Tc, X, Y, Z, mu)
    y_surface = load2(M_total, R_star, L_star, X, Y, Z, mu)

    q_inner = m_inner / M_total
    q_fit = m_fit / M_total

    sol_out = solve_ivp(
        derivs_q,
        (q_inner, q_fit),
        y_inner,
        args=(M_total, X, Y, Z, mu),
        method="LSODA",
        rtol=1.0e-8,
        atol=1.0e-10,
        max_step=(q_fit - q_inner) * 0.01,
        t_eval=np.linspace(q_inner, q_fit, 250),
    )

    N_in = 1000        # finer steps of the q for the envelop                                  
    eps  = 1.0e-100                                       
    t_eval_in = 1.0 - np.geomspace(eps, 1.0 - q_fit, N_in)
    t_eval_in = np.concatenate(([1.0], t_eval_in, [q_fit]))
    t_eval_in = np.unique(np.clip(t_eval_in, q_fit, 1.0))[::-1]

    sol_in = solve_ivp(
        derivs_q,
        (1.0, q_fit),
        y_surface,
        args=(M_total, X, Y, Z, mu),
        method="LSODA",
        rtol=1.0e-8,
        atol=1.0e-10,
        max_step=(1.0 - q_fit) * 1.0e-3,
        t_eval=t_eval_in,
    )

    m_out = sol_out.t * M_total
    y_out = sol_out.y

    m_in = sol_in.t[::-1] * M_total
    y_in = sol_in.y[:, ::-1]

    m_full = np.concatenate([m_out, m_in[1:]])
    y_full = np.concatenate([y_out, y_in[:, 1:]], axis=1)

    return m_full, y_full


def ZAMS_model(
    M_total,
    X,
    Y,
    Z,
    Pc_guess=None,
    Tc_guess=None,
    R_guess=None,
    L_guess=None,
    m_inner_frac=1.0e-10,
    m_fit_frac=0.20,
):
    mu, mu_I, mu_e = mean_molecular_weights(X, Y, Z)

    # initial guess
    if R_guess is None:
        R_guess = R_sun * (M_total / M_sun) ** 0.75

    if L_guess is None:
        L_guess = L_sun * (M_total / M_sun) ** 3.5

    if Pc_guess is None:
        Pc_guess = 2.5e17 * (M_total / M_sun) ** 2 * (R_sun / R_guess) ** 4

    if Tc_guess is None:
        Tc_guess = 1.55e7 * (M_total / M_sun) ** 0.7

    scales = {
        "Pc": Pc_guess,
        "Tc": Tc_guess,
        "R": R_guess,
        "L": L_guess,
    }

    m_inner = m_inner_frac * M_total
    m_fit = m_fit_frac * M_total

    print("Initial guesses:")
    print(f"Opacity source: {opacity_source(X, Y, Z)}")
    print(f"R  = {R_guess / R_sun:.4f} R_sun")
    print(f"L  = {L_guess / L_sun:.4f} L_sun")
    print(f"Pc = {Pc_guess:.3e} dyn cm^-2")
    print(f"Tc = {Tc_guess:.3e} K")
    print(f"m_inner = {m_inner_frac:.1e} M")
    print(f"m_fit = {m_fit_frac:.2f} M")

    #The converge requirement can be changed here
    res = root(
        shootf_residual,
        np.ones(4),
        args=(M_total, m_fit, m_inner, X, Y, Z, mu, scales),
        method="hybr",
        options={"xtol": 1.0e-5, "maxfev": 100},
    )

    Pc = res.x[0] * scales["Pc"]
    Tc = res.x[1] * scales["Tc"]
    R_star = res.x[2] * scales["R"]
    L_star = res.x[3] * scales["L"]

    m_full, y_full = final_profile(
        M_total,
        m_fit,
        m_inner,
        Pc,
        Tc,
        R_star,
        L_star,
        X,
        Y,
        Z,
        mu,
    )

    l_full, P_full, r_full, T_full = y_full

    rho_full = density(P_full, T_full, mu)
    kap_full = get_opacity(rho_full, T_full, X, Y, Z)
    eps_full = epsilon_total(rho_full, T_full, X, Y, Z, mu)
    eps_pp_full = epsilon_pp(rho_full, T_full, X, Y, Z, mu)
    eps_CNO_full = epsilon_CNO(rho_full, T_full, X, Y, Z)

    nabla_rad = (
        3.0 * kap_full * l_full * P_full
        / (16.0 * np.pi * a_rad * c_light * G * m_full * T_full**4)
    )
    

    nabla_ad = adiabatic_gradient(P_full, T_full)
    nabla = np.minimum(nabla_rad, nabla_ad)
    
    lnT_full = np.log(np.maximum(T_full, 1.0e-300))
    lnP_full = np.log(np.maximum(P_full, 1.0e-300))
    nabla_cal = np.gradient(lnT_full, lnP_full)
    
    rho_c = density(Pc, Tc, mu)

    model = {
        "success": res.success,
        "n_iter": res.nfev,
        "residual": res.fun,
        "residual_norm": np.linalg.norm(res.fun),
        "Pc_guess": Pc_guess,
        "Tc_guess": Tc_guess,
        "R_guess": R_guess,
        "L_guess": L_guess,
        "M": M_total,
        "X": X,
        "Y": Y,
        "Z": Z,
        "mu": mu,
        "mu_I": mu_I,
        "mu_e": mu_e,
        "Pc": Pc,
        "Tc": Tc,
        "R": R_star,
        "L": L_star,
        "T_eff": (L_star / (4.0 * np.pi * R_star**2 * sigma))**0.25,
        "g_surf": G * M_total / R_star**2,
        "m": m_full,
        "r": r_full,
        "rho": rho_full,
        "T": T_full,
        "P": P_full,
        "l": l_full,
        "kappa": kap_full,
        "eps": eps_full,
        "eps_pp": eps_pp_full,
        "eps_CNO": eps_CNO_full,
        "nabla_rad": nabla_rad,
        "nabla_ad": nabla_ad,
        "nabla_cal": nabla_cal,
        "nabla": nabla,
        "convective": nabla_rad > nabla_ad,
        "m_inner": m_inner,
        "m_fit": m_fit,
        "rho_c": rho_c,
    }

    print("Result:")
    if model["success"]:
        print(f"converged")
        print(f"number of iterations = {model['n_iter']}")
    if not model["success"]:
        print("not converge")
        print("Try different initial guesses")
        print(f"residual = {model['residual']}")
    print(f"R  = {model['R'] / R_sun:.4f} R_sun")
    print(f"Teff = {model['T_eff']:.0f} K")
    print(f"L  = {model['L'] / L_sun:.4f} L_sun")
    print(f"Pc = {model['Pc']:.3e} dyn cm^-2")
    print(f"Tc = {model['Tc']:.3e} K")
    print(f"mu = {model['mu']:.6f}")
    print(f"rho_c = {model['rho_c']:.3e} g cm^-3")
    print(f"g_surf = {model['g_surf']:.3e} cm s^-2")
    return model

def plot_structure(model):
    r_plot = model["r"] / model["R"]
    rho_plot = model["rho"]
    P_plot = model["P"]
    T_plot = model["T"]
    m_plot = model["m"] / model["M"]
    L_plot = model["l"] / model["L"]

    plt.figure(figsize=(6, 4))
    plt.plot(r_plot, rho_plot)
    plt.yscale("log")
    plt.xlabel(r"$r/R_\ast$")
    plt.ylabel(r"$\rho \;[\mathrm{g\,cm^{-3}}]$")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(r_plot, P_plot)
    plt.yscale("log")
    plt.xlabel(r"$r/R_\ast$")
    plt.ylabel(r"$P \;[\mathrm{dyn\,cm^{-2}}]$")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(r_plot, T_plot)
    plt.yscale("log")
    plt.xlabel(r"$r/R_\ast$")
    plt.ylabel(r"$T \;[\mathrm{K}]$")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(r_plot, m_plot)
    plt.xlabel(r"$r/R_\ast$")
    plt.ylabel(r"$M_r/M_\ast$")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(r_plot, L_plot)
    plt.xlabel(r"$r/R_\ast$")
    plt.ylabel(r"$L_r/L_\ast$")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(m_plot, L_plot)
    plt.xlabel(r"$M_r/M_\ast$")
    plt.ylabel(r"$L_r/L_\ast$")
    plt.tight_layout()
    plt.show()


def plot_opacity(model):
    r_plot = model["r"] / model["R"]
    T_plot = np.maximum(model["T"], 1.0e-30)
    m_plot = model["m"] / model["M"]
    kappa_plot = np.maximum(model["kappa"], 1.0e-30)

    plt.figure(figsize=(6, 4))
    plt.plot(r_plot, kappa_plot)
    plt.yscale("log")
    plt.xlabel(r"$r/R_\ast$")
    plt.ylabel(r"$\kappa\;[\mathrm{cm^2\,g^{-1}}]$")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(T_plot, kappa_plot)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$T\;[\mathrm{K}]$")
    plt.ylabel(r"$\kappa\;[\mathrm{cm^2\,g^{-1}}]$")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(model["m"] / model["M"], model["kappa"])
    plt.yscale("log")
    plt.xlabel(r"$M_r/M_\ast$")
    plt.ylabel(r"$\kappa\;[\mathrm{cm^2\,g^{-1}}]$")
    plt.tight_layout()
    plt.show()


def plot_nuclear_energy(model):
    r_plot = model["r"] / model["R"]
    T_plot = np.maximum(model["T"], 1.0e-99)
    logT_plot = np.log10(T_plot)

    eps_pp = np.where(model["eps_pp"] > 1.0e-99, model["eps_pp"], np.nan)
    eps_CNO = np.where(model["eps_CNO"] > 1.0e-99, model["eps_CNO"], np.nan)
    eps_tot = np.where(model["eps"] > 1.0e-99, model["eps"], np.nan)

    idx_T = np.argsort(logT_plot)

    plt.figure(figsize=(6, 4))
    plt.plot(logT_plot[idx_T], eps_pp[idx_T], label="pp")
    plt.plot(logT_plot[idx_T], eps_CNO[idx_T], label="CNO")
    plt.plot(logT_plot[idx_T], eps_tot[idx_T], color="lightcoral", linestyle=":", label="total")
    plt.yscale("log")
    plt.xlabel(r"$\log T$")
    plt.ylabel(r"$\epsilon\;[\mathrm{erg\,g^{-1}\,s^{-1}}]$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(r_plot, eps_pp, label="pp")
    plt.plot(r_plot, eps_CNO, label="CNO")
    plt.plot(r_plot, eps_tot, linestyle=":", color="lightcoral", label="total")
    plt.yscale("log")
    plt.xlabel(r"$r/R$")
    plt.ylabel(r"$\epsilon\;[\mathrm{erg\,g^{-1}\,s^{-1}}]$")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_temperature_gradients(model):
    r_plot = model["r"] / model["R"]
    nabla_ad = model["nabla_ad"]
    nabla_rad = model["nabla_rad"]
    nabla_actual = model["nabla"]
    nabla_cal = model["nabla_cal"]

    plt.figure(figsize=(6, 4))
    plt.plot(r_plot, nabla_actual, linewidth=2.0, label=r"$\nabla$")
    plt.plot(r_plot, nabla_ad, linewidth=2.0, linestyle="--", label=r"$\nabla_{\rm ad}$")
    plt.plot(r_plot, nabla_rad, linewidth=2.0, linestyle=":", label=r"$\nabla_{\rm rad}$")
    #plt.plot(r_plot, nabla_cal, linestyle=":", linewidth=2.5, label=r"$\nabla_{calculated back}$")

    plt.xlabel(r"$r/R_\ast$")
    plt.ylabel(r"$\nabla$")
    plt.ylim(0, min(1.0, 1.1 * np.nanmax(nabla_rad[np.isfinite(nabla_rad)])))
    plt.legend()
    plt.tight_layout()
    plt.show()


def export_structure_table(model, filename=None):

    def fmt_for_filename(value, ndigits=2):

        return f"{value:.{ndigits}f}".replace(".", "p").replace("-", "m")
    m = np.asarray(model["m"], dtype=float)
    r = np.asarray(model["r"], dtype=float)
    rho = np.asarray(model["rho"], dtype=float)
    T = np.asarray(model["T"], dtype=float)
    P = np.asarray(model["P"], dtype=float)
    l = np.asarray(model["l"], dtype=float)
    eps = np.asarray(model["eps"], dtype=float)
    kappa = np.asarray(model["kappa"], dtype=float)

    nabla_ad = np.asarray(model["nabla_ad"], dtype=float)
    nabla_rad = np.asarray(model["nabla_rad"], dtype=float)

    idx = np.argsort(m)

    m = m[idx]
    r = r[idx]
    rho = rho[idx]
    T = T[idx]
    P = P[idx]
    l = l[idx]
    eps = eps[idx]
    kappa = kappa[idx]
    nabla_ad = nabla_ad[idx]
    nabla_rad = nabla_rad[idx]

    lnT = np.log(np.maximum(T, 1.0e-300))
    lnP = np.log(np.maximum(P, 1.0e-300))
    nabla_dlnT_dlnP = np.gradient(lnT, lnP, edge_order=2)

    shell_type = np.where(nabla_rad > nabla_ad, "convective", "radiative")

    if filename is None:
        M_over_Msun = model["M"] / M_sun
        X = model["X"]
        Y = model["Y"]
        Z = model["Z"]

        filename = (
            "zams_structure_"
            f"M{fmt_for_filename(M_over_Msun, 2)}_"
            f"X{fmt_for_filename(X, 2)}_"
            f"Y{fmt_for_filename(Y, 2)}_"
            f"Z{fmt_for_filename(Z, 2)}.txt"
        )

    table = pd.DataFrame({
        "m": m,
        "r": r,
        "rho": rho,
        "T": T,
        "P": P,
        "l": l,
        "eps": eps,
        "kappa": kappa,
        "nabla_ad": nabla_ad,
        "nabla_rad": nabla_rad,
        "nabla_dlnT_dlnP": nabla_dlnT_dlnP,
        "shell_type": shell_type,
    })

    # Header for the txt file
    header = (
        f"M = {model['M'] / M_sun:.3e} M_sun, "
        f"X = {model['X']:.2e}, Y = {model['Y']:.2e}, Z = {model['Z']:.2e}\n"
        f"Output model: R = {model['R'] / R_sun:.8e} R_sun, "
        f"L = {model['L'] / L_sun:.4e} L_sun, "
        f"Teff = {model['T_eff']:.4e} K, "
        f"Pc = {model['Pc']:.4e} dyn cm^-2, "
        f"Tc = {model['Tc']:.4e} K\n"
    )

    with open(filename, "w") as f:
        for line in header.split("\n"):
            if line.strip() != "":
                f.write("# " + line + "\n")

        table.to_csv(
            f,
            sep=" ",
            index=False,
            float_format="%.4e",
        )

    print(f"saved to: {filename}")

    return table


#sample input
M_total = 1.2 * M_sun
X = 0.72
Z = 0.01
Y = 1.0 - X - Z

# Please change "table_filename" function in opacity_table.py if your composition is not in two decimals

model = ZAMS_model(
    M_total=M_total,
    X=X,
    Y=Y,
    Z=Z,
    Pc_guess=2.0e17,
    Tc_guess=1.5e7,
    R_guess=1 * R_sun,
    L_guess=2 * L_sun,
    m_fit_frac=0.40,
)

plot_structure(model)

plot_opacity(model)

plot_nuclear_energy(model)

plot_temperature_gradients(model)

structure_table = export_structure_table(model)


