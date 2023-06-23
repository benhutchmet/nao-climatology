# Import the relevant modules
import numpy as np
import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import pearsonr, mstats, ttest_rel, ttest_ind, ttest_1samp
from sklearn.utils import resample
# import the datetime library
from datetime import datetime

# Import dictionaries
import dictionaries as dic

def load_data(path, variable):
    """
    Load data from a NetCDF file and select a specific variable.
    
    Parameters
    ----------
    path : str
        Path to the NetCDF file.
    variable : str
        Name of the variable to select.
        
    Returns
    -------
    data : xarray DataArray
        The loaded data for the selected variable.
    """
    
    # Load the NetCDF file using xarray
    ds = xr.open_dataset(path, chunks={'time': 100})
    
    # Select the variable using its name
    data = ds[variable]

    return data

def NAO_index(psl_data, azores_grid, iceland_grid):
    """
    This function calculates the normalised NAO index.
    
    Parameters
    ----------
    psl_data : xarray DataArray
        The sea level pressure data.
    azores_grid : dict
        The grid box for the Azores region, defined as a dictionary with keys 'lon1', 'lon2', 'lat1', and 'lat2'.
    iceland_grid : dict
        The grid box for the Iceland region, defined as a dictionary with keys 'lon1', 'lon2', 'lat1', and 'lat2'.

    Returns
    -------
    NAO_index : array
        The normalised NAO index.
    """
    
    # Extract the psl values for the Azores grid box
    azores_psl = psl_data.sel(longitude=slice(azores_grid['lon1'], azores_grid['lon2']),
                              latitude=slice(azores_grid['lat1'], azores_grid['lat2']))
    
    # Extract the psl values for the Iceland grid box
    iceland_psl = psl_data.sel(longitude=slice(iceland_grid['lon1'], iceland_grid['lon2']),
                               latitude=slice(iceland_grid['lat1'], iceland_grid['lat2']))
    
    # Calculate the NAO index as the difference between Azores and Iceland
    NAO_index = azores_psl.mean(dim=('longitude', 'latitude')) \
        - iceland_psl.mean(dim=('longitude', 'latitude'))
    
    # Normalize the NAO index
    norm_NAO_index = (NAO_index - NAO_index.mean()) / NAO_index.std()
    
    return norm_NAO_index

# Write a function that will choose anomalies of +/- 1 standard deviation
# from the normalised NAO index
# + 1 standard deviation - positive NAO anomalies
# - 1 standard deviation - negative NAO anomalies
# and return the indices and dates of these anomalies
def select_NAO_anomalies(norm_NAO_index):
    """
    Select anomalies of +/- 1 standard deviation from the normalized NAO index.
    
    Parameters
    ----------
    norm_NAO_index : array-like
        The normalized NAO index.
        
    Returns
    -------
    pos_anomalies : array-like
        Positive anomalies of +/- 1 standard deviation from the normalized NAO index.
    pos_anomaly_indices : array-like
        Indices of the selected positive anomalies.
    pos_anomaly_dates : array-like
        Dates corresponding to the selected positive anomalies.
    neg_anomalies : array-like
        Negative anomalies of +/- 1 standard deviation from the normalized NAO index.
    neg_anomaly_indices : array-like
        Indices of the selected negative anomalies.
    neg_anomaly_dates : array-like
        Dates corresponding to the selected negative anomalies.
    """
    
    # Calculate the mean and standard deviation of the normalized NAO index
    mean = norm_NAO_index.mean()
    std = norm_NAO_index.std()
    
    # Define the upper and lower thresholds for anomaly selection
    upper_threshold = mean + std
    lower_threshold = mean - std
    
    # Select the positive and negative anomalies based on the upper and lower thresholds
    pos_anomalies = norm_NAO_index[norm_NAO_index >= upper_threshold]
    neg_anomalies = norm_NAO_index[norm_NAO_index <= lower_threshold]
    
    # Get the indices of the selected positive and negative anomalies
    pos_anomaly_indices = pos_anomalies.index.values
    neg_anomaly_indices = neg_anomalies.index.values
    
    # Convert the indices to dates for positive and negative anomalies
    pos_anomaly_dates = pd.to_datetime(pos_anomaly_indices)
    neg_anomaly_dates = pd.to_datetime(neg_anomaly_indices)
    
    return pos_anomalies, pos_anomaly_indices, pos_anomaly_dates, neg_anomalies, neg_anomaly_indices, neg_anomaly_dates

# Select the time series of the other variable with only the selected positive and negative anomaly indices/dates
def select_anomaly_time_series(pos_anomaly_indices, pos_anomaly_dates, neg_anomaly_indices, neg_anomaly_dates, other_variable_data):
    """
    Select time series of the other variable with only the selected positive and negative anomaly indices/dates.
    
    Parameters
    ----------
    pos_anomaly_indices : array-like
        Indices of the selected positive anomalies.
    pos_anomaly_dates : array-like
        Dates corresponding to the selected positive anomalies.
    neg_anomaly_indices : array-like
        Indices of the selected negative anomalies.
    neg_anomaly_dates : array-like
        Dates corresponding to the selected negative anomalies.
    other_variable_data : xarray DataArray
        The loaded data for the other variable (e.g., surface wind or temperature).
        
    Returns
    -------
    pos_anomaly_time_series : xarray DataArray
        The time series of the other variable with only the selected positive anomaly indices/dates.
    neg_anomaly_time_series : xarray DataArray
        The time series of the other variable with only the selected negative anomaly indices/dates.
    """
    
    # Select the time series of the other variable using the positive and negative anomaly indices/dates
    pos_anomaly_time_series = other_variable_data.sel(time=pos_anomaly_indices)
    neg_anomaly_time_series = other_variable_data.sel(time=neg_anomaly_indices)
    
    # Assign the corresponding dates to the time series
    pos_anomaly_time_series['time'] = pos_anomaly_dates
    neg_anomaly_time_series['time'] = neg_anomaly_dates
    
    return pos_anomaly_time_series, neg_anomaly_time_series

def constrain_to_north_atlantic(anomaly_time_series, lat_range, lon_range):
    """
    Constrain the anomaly time series to the North Atlantic region and take the time mean.
    
    Parameters
    ----------
    anomaly_time_series : xarray DataArray
        Anomaly time series of the variable.
    lat_range : slice
        Latitude range for the North Atlantic region.
    lon_range : slice
        Longitude range for the North Atlantic region.
        
    Returns
    -------
    time_mean_constrained_time_series : xarray DataArray
        Time-mean anomaly time series constrained to the North Atlantic region.
    """
    
    # Select the time series within the North Atlantic region
    constrained_time_series = anomaly_time_series.sel(latitude=lat_range, longitude=lon_range)
    
    # Take the time mean of the constrained time series
    time_mean_constrained_time_series = constrained_time_series.mean(dim='time')
    
    return time_mean_constrained_time_series

# Plot the data
def plot_time_mean_constrained(time_mean_constrained, variable):
    """
    Plot the time-mean-constrained time series on a map using cartopy.
    
    Parameters
    ----------
    time_mean_constrained : xarray DataArray
        Time-mean-constrained time series of the variable.
    variable : str
        Name of the variable to use for the colorbar label.
    """
    
    # Create a figure and an axes with a specific projection (e.g., Plate Carree)
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Plot the time-mean-constrained time series as a filled contour
    c = ax.contourf(time_mean_constrained.longitude, time_mean_constrained.latitude,
                    time_mean_constrained, transform=ccrs.PlateCarree(), cmap='coolwarm')
    
    # Add coastlines and gridlines
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    
    # Add a colorbar
    cbar = fig.colorbar(c, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label(variable)
    
    # Set the title
    ax.set_title('Time-Mean-Constrained Time Series')
    
    # Show the plot
    plt.show()
