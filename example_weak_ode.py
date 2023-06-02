



import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
import pysindy as ps
from weak_pde_library import WeakPDELibrary



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
dt = 0.002
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
    K=100,
)





"""
optimizer = ps.SR3(threshold=0.1, 
            thresholds=None, nu=1.0, tol=1e-05, 
            thresholder='L0', trimming_fraction=0.0, 
            trimming_step_size=1.0, max_iter=30, 
            fit_intercept=False, copy_X=True, 
            initial_guess=None, 
            normalize_columns=False, 
            verbose=False)
"""


# Instantiate and fit the SINDy model with the integral of u_dot
optimizer = ps.SR3(
    threshold=0.05, thresholder="l1", max_iter=1000, normalize_columns=True, tol=1e-1
)
model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
model.fit(u_train)
model.print()

optimizer = ps.SR3(threshold=0.1, 
            thresholds=None, nu=1.0, tol=1e-05, 
            thresholder='L0', trimming_fraction=0.0, 
            trimming_step_size=1.0, max_iter=30, 
            fit_intercept=False, copy_X=True, 
            initial_guess=None, 
            normalize_columns=False, 
            verbose=False)




# Instantiate and fit the SINDy model with u_dot
u_dot = ps.FiniteDifference()._differentiate(u_train, t=dt)
model = ps.SINDy()
model.fit(u_train, x_dot=u_dot, t=dt)
model.print()

#######################################
#######################################
#######################################
##################### cubic oscilator




print("===================")


dt = 0.002
t_train = np.arange(0, 10, dt)
t_train_span = (t_train[0], t_train[-1])
u0_train = [-2, 2]
u_train = solve_ivp(
    cubic_oscillator, t_train_span, u0_train, t_eval=t_train, **integrator_keywords
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
    K=100,
)




"""
# Instantiate and fit the SINDy model with the integral of u_dot
optimizer = ps.SR3(
    threshold=0.05, thresholder="l1", max_iter=1000, normalize_columns=True, tol=1e-1
)
model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
model.fit(u_train)
model.print()
"""
