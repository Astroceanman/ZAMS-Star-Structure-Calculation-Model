# ZAMS-Star-Structure-Calculation-Model

## Overview
This model calculate the structure of zero-age main-sequence (ZAMS) stars with different masses and different compositions.

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

If you are using Jupyter Notebook or other similar environments, the use of the following is recommended for better plot viewing.

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

The input of mass and composition should be in the format as follow:

```
M_total = 1.2 * M_sun
X = 0.72
Z = 0.01
Y = 1.0 - X - Z
```

X is the mass fraction of hydrogen, Y is the mass fraction of helium, and Z is the mass fraction of metals. Notably, the sum of X, Y, and Z should equal to zero to ensure the physical valid model.

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

## Algorithm

## Results


## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.
