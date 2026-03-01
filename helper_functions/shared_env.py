# The shared imports for all the models.
# Standard Libraries
import sys
import os
import time
import random
import pickle
import itertools
import statistics

# Scientific Stack
import numpy as np
from numpy import loadtxt
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import stats
from tqdm import tqdm

# Notebook/UI
import ipywidgets as widgets
from IPython.display import display, clear_output

# Functions
import importlib
sys.path.append(os.path.abspath(os.path.join('..')))

try:
    import helper_functions.base_visualisations as bv    
    importlib.reload(bv)
    from helper_functions.base_visualisations import *
except ModuleNotFoundError:
    # This fall-back handles cases where you are running inside the folder
    import base_visualisations as bv
    importlib.reload(bv)
    from base_visualisations import *
    
# Configuration
np.random.seed(0)
plt.style.use(os.path.join(os.path.dirname(__file__), '../style_sheet.mplstyle'))
