# Import the relevant modules
import numpy as np
import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mstats, ttest_rel, ttest_ind, ttest_1samp
from sklearn.utils import resample
# import the datetime library
from datetime import datetime

# Import dictionaries
import dictionaries as dic

# Define a function
# Which calculates the normalised NAO index
# As the difference between two grid boxes in the North Atlantic
# From the psl data
def NAO_index(psl_data_path):
    """
    This function calculates the normalised NAO index.
    
    Parameters
    ----------
    psl_data_path : str
        The path to the psl data.
        
    Returns
    -------
    NAO_index : array
        The normalised NAO index.
    """

#     # if the location is azores
# if [ $location == "azores" ]; then
#     # set the dimensions of the gridbox
#     lon1=-28
#     lon2=-20
#     lat1=36
#     lat2=40
#     # set the grid
#     grid=$azores_grid
# elif [ $location == "iceland" ]; then
#     # set the dimensions of the gridbox
#     lon1=-25
#     lon2=-16
#     lat1=63
#     lat2=70
#     # set the grid
#     grid=$iceland_grid
# else
#     echo "Location must be azores or iceland"
#     exit 1
# fi


