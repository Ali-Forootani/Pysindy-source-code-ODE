# Pysindy-source-code-ODE

I was asked to use weak sindy algorithm to identify lorenz system from data. 
however the original pysindy library had some bugs, so it obliged me to go to its details and
excavate the library. if you want to run one simulation you can run `example_weak_ode_2.p`.

# What is PySINDy?

PySINDy is a Python package for the sparse identification of nonlinear dynamical systems (SINDy). SINDy is a data-driven method that can discover the governing equations or dynamics from time-series data. It is particularly useful when working with complex systems that lack explicit models or are difficult to model accurately.

This package provides a user-friendly interface for using SINDy in Python. It supports both continuous and discrete-time data, and offers various algorithms for model discovery and equation learning. PySINDy also includes additional features such as noise handling, cross-validation, custom library functions, and model selection.

# Let's see the example
 Here's a breakdown of the code:

    1. Importing the required libraries and modules:
    `python import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
import pysindy as ps
from weak_pde_library import WeakPDELibrary
from sr3_class import SR3
import inspect
'
