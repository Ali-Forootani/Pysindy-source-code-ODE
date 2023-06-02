#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:17:19 2023

@author: forootani
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
import pysindy as ps
from weak_pde_library import WeakPDELibrary

from sr3_class import SR3

import inspect

# Ignore matplotlib deprecation warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Seed the random number generators for reproducibility
np.random.seed(100)

# integration keywords for solve_ivp, typically needed for chaotic systems
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12



def lorenz(t, x, sigma=10, beta=2.66667, rho=28):
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]


def cubic_oscillator(t, x, p=[-0.1, 2, -2, -0.1]):
    return [p[0] * x[0] ** 3 + p[1] * x[1] ** 3, p[2] * x[0] ** 3 + p[3] * x[1] ** 3]




# Generate measurement data
dt = 0.0005
t_train = np.arange(0, 10, dt)
t_train_span = (t_train[0], t_train[-1])
u0_train = [-8, 8, 27]
u_train = solve_ivp(
    lorenz, t_train_span, u0_train, t_eval=t_train, **integrator_keywords
).y.T





# Instantiate and fit the SINDy model with u_dot
u_dot = ps.FiniteDifference()._differentiate(u_train, t=dt)
model = ps.SINDy()
model.fit(u_train, x_dot=u_dot, t=dt)
model.print()

# Define weak form ODE library
# defaults to derivative_order = 0 if not specified,
# and if spatial_grid is not specified, defaults to None,
# which allows weak form ODEs.
library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]
ode_lib = WeakPDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    spatiotemporal_grid=t_train,
    is_uniform=True,
    K=20000,
)


"""
signature = inspect.signature(SR3)
# Get the parameters of the constructor
parameters = signature.parameters

# Iterate over the parameters and print their names and default values (if any)
for param_name, param in parameters.items():
    print("Parameter:", param_name)
    print("Default value:", param.default)
    print("---")

"""




print("88888888888888888888888888888888888888")
print("88888888888888888888888888888888888888")
print("88888888888888888888888888888888888888")



optimizer = SR3(
    threshold=0.1, thresholder="l0", normalize_columns=True, max_iter=1000, tol=1e-8
)


model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
model.fit(u_train)
model.print()




print("88888888888888888888888888888888888888")
print("88888888888888888888888888888888888888")
print("88888888888888888888888888888888888888")


import numpy as np
from scipy.integrate import odeint
from pysindy import SINDy
#from pysindy.optimizers import SR3
lorenz = lambda z,t : [10 * (z[1] - z[0]),
                        z[0] * (28 - z[2]) - z[1],
                   z[0] * z[1] - 8 / 3 * z[2]]
t = np.arange(0, 10, .025)
x = odeint(lorenz, [-8, 8, 27], t)
opt = SR3(threshold=0.1, nu=1, max_iter=2000)
model = SINDy(optimizer=opt)
model.fit(x, t=t[1] - t[0])
model.print()
#x0' = -10.004 1 + 10.004 x0
#x1' = 27.994 1 + -0.993 x0 + -1.000 1 x1
#x2' = -2.662 x1 + 1.000 1 x0
















