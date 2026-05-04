# ZAMS-Star-Structure-Calculation-Model

## Overview
This model calculate the structure of zero-age main-sequence (ZAMS) stars with different masses and different compositions. Our calculation model solves the luminosity, $L$, pressure, $P$, radius, $r$, and temperature, $T$, of ZAMS stars by using the shooting method. The model is constructed under the assumptions of spherical symmetry, hydrostatic equilibrium, thermal equilibrium, homogeneous composition, a fully ionized ideal gas, and no rotation.

## Getting Started

Ensure that Python 3.7 or higher is installed. The code requires the following Python packages:

• numpy

• scipy

• matplotlib

• pandas

• os

• math

Install the required packages using the command:

```
pip install numpy scipy matplotlib pandas
```

If running the code in Jupyter Notebook or other similar environments, you may optionally enable interactive figures.

```
%matplotlib widget
```

This can be installed using the command:

```
pip install ipympl
```

## Input of the Model

The model requires the input of the following quantities:

• Mass of the star

• Chemical composition of the star

• Initial guess of central pressure

• Initial guess of central temperature

• Initial guess of radius

• Initial guess of luminosity

• Mass fraction of the fitting point

All the values used in the model should be in the cgs unit.

The input of mass and composition should be in the format as follow:

```
M_total = 1.2 * M_sun
X = 0.72
Z = 0.01
Y = 1.0 - X - Z
```

X is the mass fraction of hydrogen, Y is the mass fraction of helium, and Z is the mass fraction of metals. Notably, the sum of X, Y, and Z should equal to 1.0 to ensure the physical valid model.

If the initial gusses is None, then the model will use the apprximated initial gusses from mass-scaling guesses. 

The choice of fitting point can affect the runtime and the number of iterations required for convergence. The maximum iterations is set as 100 in this model, and can be changed by changing ```maxfev```. The threshold of convergence is set as the residual smaller than $10^{-5}$ and can be changed by changing ```xtol```.

```
res = root(
        shootf_residual,
        np.ones(4),
        args=(M_total, m_fit, m_inner, X, Y, Z, mu, scales),
        method="hybr",
        options={"xtol": 1.0e-5, "maxfev": 100},
    )
```

The adjustment of initial gusseses and fitting point is recommended for shorter runtime and fewer iterations. 

The model prints the number of iterations required for convergence.

## Import Statements
The script begins with necessary import statements for data manipulation, numerical operations, plotting, and statistical analysis:

```
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
%matplotlib widget
```

## Opacity Table

The opacity table with different compositions should be downloaded from [The Opacity Project](https://cds.unistra.fr/topbase/home.html).

For the model, the table should be downloaded using the option "RMOs for a single chemical mixture," and with the table format $\{\log_{10}(T),\log_{10}(R)\}$. 

The downloaded opacity table should be renamed according to the format `op_X0p72_Z0p01_logT_logR.txt`, where `X0p72` and `Z0p01` indicate the adopted hydrogen and metal mass fractions, \(X=0.72\) and \(Z=0.01\), respectively.

All opacity tables should be located in the `opacity_tables` folder.

If the opacity table for certain composition is not available in the `opacity_tables` folder, the model will automatically approximate the opacity by using $\kappa=\kappa_{\rm es}+\kappa_{\rm bf}+\kappa_{\rm ff}+\kappa_{\rm H^-}$.

Notably, the approximation of opacity could be inaccurate, so the downloaded opacity table is recommended.

The model prints whether it is using an opacity table and, if so, which table is used.

The opacity tables are read by `opacity_table.py`.

In `stellar_structure_calculation.py`, the model uses the opacity tables in the calculation by:

```
from opacity_table import opacity, opacity_source, opacity_grid
```

## Algorithm

The model solves the stellar structure of a ZAMS star by using the shooting method. The four dependent variables are luminosity \(l\), pressure \(P\), radius \(r\), and temperature \(T\).

The stellar structure equations are:

$$
\frac{dl}{dm} = \epsilon
$$

$$
\frac{dP}{dm} = -\frac{Gm}{4\pi r^4}
$$

$$
\frac{dr}{dm} = \frac{1}{4\pi r^2\rho}
$$

$$
\frac{dT}{dm} = -\frac{GmT}{4\pi r^4P}\nabla
$$

The calculation process follow:

(1) Calculating the mean molecular weight by using the input X, Y, and Z.

(2) Define the equation of state including ideal-gas pressure and radiation pressure.

(3) Calculate the density from ideal gas law.

(4) Read opacity table or calculate opacity.

(5) Calculate nuclear energy generation of the proton-proton chain and CNO cycle.

(6) Calculate the radiative and adiabatic temperature gradients, and the actual temperature gradient is approximated as $\nabla = \min(\nabla_{\rm rad}, \nabla_{\rm ad})$.

(7) Define the central and surface boundary conditions.

Notably, for the central boundary condition, the mass is selected as $10^{-10}$ $M_*$ to avoid invalid values. This boundary can be change by changing ```m_inner_frac``` in

```
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
```

(8) The model integrates outward from the center and inward from the surface to a selected fitting point, as explained in the "Input" section. The four shooting parameters are $P_{\rm c}, \quad T_{\rm c}, \quad R_\ast, \quad L_\ast.$

(9) The model calculates the residual at the fitting point, defined as the logarithmic difference between the outward and inward solutions. Then the model will adjust the shooting parameters until the residual of all four dependent variables are within $10^{-5}$. The adjustment of this threshold is explain in the "Input" section

(10) After converge, the model will recomputes the outward and inward integrations using the final shooting parameters and combines the two solutions into one stellar structure profile.


## Results

The model prints the initial guesses and key calculated stellar structure quantities in the cgs unit.

The output follows the format:

```
Initial guesses:
Opacity source: OP table: op_X0p72_Z0p01_logT_logR.txt
R  = 1.0000 R_sun
L  = 2.0000 L_sun
Pc = 2.000e+17 dyn cm^-2
Tc = 1.500e+07 K
m_inner = 1.0e-10 M
m_fit = 0.40 M
Result:
converged
number of iterations = 31
R  = 1.2784 R_sun
Teff = 6358 K
L  = 2.4097 L_sun
Pc = 2.104e+17 dyn cm^-2
Tc = 1.636e+07 K
mu = 0.606717
rho_c = 9.450e+01 g cm^-3
g_surf = 2.012e+04 cm s^-2
```

The model also plots the radial profiles of mass, density, temperature, pressure, luminosity, nuclear energy generation rate, opacity, temperature gradients.

The model also stores the calculated results in a `.txt` file. The file's name follows the format of `zams_structure_M1p20_X0p72_Y0p27_Z0p01.txt`, where `M1p20`, `X0p72`, `Y0p27`, `Z0p01` means $M_* = M_\odot$, $X = 0.72$, $Y = 0.27$, $Z = 0.01$, respectively.

The table includes mass, radius, density, temperature, pressure, luminosity, total nuclear energy generation, opacity, radiative and adiabatic temperature gradients, actual temperature gradient, and the radiative or convective nature of each shell.

A example calculation of the ZAMS star with $M_*=1.2\M_\odot,\X=0.72,\Y=0.27$, and $Z=0.01$ is used in the code and the results table is uploaded.


## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.
