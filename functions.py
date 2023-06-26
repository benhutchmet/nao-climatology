# Import the relevant modules
import numpy as np
import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.util as cutil
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

def test_constrain(psl_data):
    """
    Test function
    """
    azores_psl=psl_data.sel(lon=slice(152,160),lat=slice(40,36))

    print(np.shape(azores_psl))
    print(azores_psl.compute())
    

# New implementation of the function to calculate the NAO index
# Calculates NAO by season
# Seems to be working
def NAO_index(psl_data, azores_grid, iceland_grid):
    """
    This function calculates the normalized NAO index as the average of December of the current year and January, February, and March of the next year.
    
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
    norm_NAO_index : xarray DataArray
        The normalized NAO index.
    """
    
    try:

        # Shift the time index back by 4 months
        shifted_data = psl_data.roll(time=-4)

        print("shifted data", shifted_data)
        
        # Group the data by year and calculate the mean
        yearly_mean = shifted_data.groupby('time.year').mean(dim='time')

        print("yearly mean", yearly_mean)

        # Assign datetime objects to each year in the dataset
        # yearly_mean = yearly_mean.assign_coords(time=[f"{year}-12-01" for year in yearly_mean.year.values])

        # print("yearly mean", yearly_mean)
        
        # Extract the psl values for the Azores grid box
        azores_psl = yearly_mean.sel(
            lon=slice(azores_grid['lon1'], azores_grid['lon2']),
            lat=slice(azores_grid['lat1'], azores_grid['lat2'])
        )

        # Extract the psl values for the Iceland grid box
        iceland_psl = yearly_mean.sel(
            lon=slice(iceland_grid['lon1'], iceland_grid['lon2']),
            lat=slice(iceland_grid['lat1'], iceland_grid['lat2'])
        )

        # Calculate the NAO index as the difference between Azores and Iceland
        NAO_index = azores_psl.mean(dim=("lon", "lat")) - iceland_psl.mean(dim=("lon", "lat"))
        
        # Print the actual values of NAO_index, NAO_index.mean(), and NAO_index.std()
        # print("NAO_index values:", NAO_index.values)
        # print("NAO_index mean:", NAO_index.mean().values)
        # print("NAO_index_sd:", NAO_index.std().values)
        
        # Normalize the NAO index
        norm_NAO_index = (NAO_index - NAO_index.mean()) / NAO_index.std()
        
        return norm_NAO_index

    except Exception as e:
        print("Error occurred during NAO index calculation:")
        print(e)
        return None



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
    norm_NAO_index : xarray DataArray
        The normalized NAO index.
        
    Returns
    -------
    pos_anomalies : xarray DataArray
        Positive anomalies of +/- 1 standard deviation from the normalized NAO index.
    pos_anomaly_indices : numpy array
        Indices of the selected positive anomalies.
    pos_anomaly_dates : pandas DatetimeIndex
        Dates corresponding to the selected positive anomalies.
    neg_anomalies : xarray DataArray
        Negative anomalies of +/- 1 standard deviation from the normalized NAO index.
    neg_anomaly_indices : numpy array
        Indices of the selected negative anomalies.
    neg_anomaly_dates : pandas DatetimeIndex
        Dates corresponding to the selected negative anomalies.
    """
    
    # Calculate the mean and standard deviation of the normalized NAO index
    mean = norm_NAO_index.mean()
    std = norm_NAO_index.std()
    
    # Define the upper and lower thresholds for anomaly selection
    upper_threshold = mean + std
    lower_threshold = mean - std
    
    # Select the positive and negative anomalies based on the upper and lower thresholds
    pos_anomalies = norm_NAO_index.where(norm_NAO_index >= upper_threshold, drop=True)
    neg_anomalies = norm_NAO_index.where(norm_NAO_index <= lower_threshold, drop=True)
    
    # Check if positive anomalies exist
    if pos_anomalies.size == 0:
        raise ValueError("No positive anomalies found.")
    
    # Check if negative anomalies exist
    if neg_anomalies.size == 0:
        raise ValueError("No negative anomalies found.")
    
    # Get the indices of the selected positive and negative anomalies
    pos_anomaly_indices = pos_anomalies.year.values
    neg_anomaly_indices = neg_anomalies.year.values
    
    # Convert the indices to dates for positive and negative anomalies
    pos_anomaly_dates = pd.to_datetime(pos_anomaly_indices)
    neg_anomaly_dates = pd.to_datetime(neg_anomaly_indices)
    
    return pos_anomalies, pos_anomaly_indices, pos_anomaly_dates, neg_anomalies, neg_anomaly_indices, neg_anomaly_dates


# Select the time series of the other variable with only the selected positive and negative anomaly indices/dates
# This needs to be tested
def select_anomaly_time_series(pos_anomaly_indices, pos_anomaly_dates, neg_anomaly_indices, neg_anomaly_dates, other_variable_data):
    """
    Select time series of the other variable with only the selected positive and negative anomaly indices/dates.
    Calculate the anomalies of the positive and negative time series by removing the overall time-mean of the other variable.
    
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
        The time series of the other variable with only the selected positive anomaly indices/dates, with the overall time-mean removed.
    neg_anomaly_time_series : xarray DataArray
        The time series of the other variable with only the selected negative anomaly indices/dates, with the overall time-mean removed.
    """
    
    # Calculate the overall time-mean of the other variable
    time_mean = other_variable_data.mean(dim='time')

    # Shift the time series of the other variable back by -4
    # In the same way as the NAO
    # to give the same winters
    other_variable_data = other_variable_data.roll(time=-4)

    # group by year and take the mean
    other_variable_data = other_variable_data.groupby('time.year').mean(dim='time')
    
    # Select the time series of the other variable using the positive and negative anomaly indices/dates
    pos_anomaly_time_series = other_variable_data.sel(time=pos_anomaly_indices)
    neg_anomaly_time_series = other_variable_data.sel(time=neg_anomaly_indices)
    
    # Remove the overall time-mean from the positive and negative time series to calculate the anomalies
    pos_anomaly_time_series = pos_anomaly_time_series - time_mean
    neg_anomaly_time_series = neg_anomaly_time_series - time_mean
    
    # Assign the corresponding dates to the time series
    pos_anomaly_time_series['time'] = pos_anomaly_dates
    neg_anomaly_time_series['time'] = neg_anomaly_dates
    
    return pos_anomaly_time_series, neg_anomaly_time_series

# Constrain the anomaly time series to the North Atlantic region
# and take the time mean
def constrain_to_north_atlantic(anomaly_time_series, north_atlantic_grid):
    """
    Constrain the anomaly time series to the North Atlantic region and take the time mean.

    Parameters
    ----------
    anomaly_time_series : xarray DataArray
        Anomaly time series of the variable.
    north_atlantic_grid : dict
        Dictionary defining the North Atlantic grid with 'lon1', 'lon2', 'lat1', and 'lat2' keys.

    Returns
    -------
    time_mean_constrained_time_series : xarray DataArray
        Time-mean anomaly time series constrained to the North Atlantic region.
    """

    lon1, lon2 = north_atlantic_grid['lon1'], north_atlantic_grid['lon2']
    lat1, lat2 = north_atlantic_grid['lat1'], north_atlantic_grid['lat2']

    # Add a cyclic point to the dataset
    # anomaly_time_series, lon = cutil.add_cyclic_point(anomaly_time_series, coord=anomaly_time_series['longitude'])

    # Roll longitude to 0-360 if necessary
    if (anomaly_time_series.coords['longitude'] < 0).any():
        anomaly_time_series.coords['longitude'] = np.mod(anomaly_time_series.coords['longitude'], 360)
        anomaly_time_series = anomaly_time_series.sortby(anomaly_time_series.longitude)

    # Select the time series within the North Atlantic region
    if lon1 < lon2:
        constrained_time_series = anomaly_time_series.sel(
            longitude=slice(lon1, lon2),
            latitude=slice(lat1, lat2))
    else:
        # If the region wraps around the prime meridian, select two slices and concatenate
        constrained_time_series = xr.concat([
            anomaly_time_series.sel(
                longitude=slice(lon1, 360),
                latitude=slice(lat1, lat2)),
            anomaly_time_series.sel(
                longitude=slice(0, lon2),
                latitude=slice(lat1, lat2))
        ], dim='latitude')

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


# Plot the data as 2 subplots with a constant colourbar
# This needs to be tested
def plot_time_mean_constrained(time_mean_constrained_pos, time_mean_constrained_neg, variable):
    """
    Plot the time-mean-constrained time series on a map using cartopy.

    Parameters
    ----------
    time_mean_constrained_pos : xarray DataArray
        Time-mean-constrained time series of the positive NAO anomalies of the variable.
    time_mean_constrained_neg : xarray DataArray
        Time-mean-constrained time series of the negative NAO anomalies of the variable.
    variable : str
        Name of the variable to use for the colorbar label.
    """
    
    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Plot the positive anomalies on the left
    pos_plot = time_mean_constrained_pos.plot(ax=ax1, cmap='RdBu_r', vmin=-2, vmax=2, add_colorbar=False)
    ax1.set_title('Positive NAO Anomalies')
    
    # Plot the negative anomalies on the right
    neg_plot = time_mean_constrained_neg.plot(ax=ax2, cmap='RdBu_r', vmin=-2, vmax=2, add_colorbar=False)
    ax2.set_title('Negative NAO Anomalies')
    
    # Add a common colorbar
    cbar = fig.colorbar(pos_plot, ax=[ax1, ax2], orientation='horizontal', pad=0.05)
    cbar.set_label(variable)
    
    # Add coastlines and gridlines
    for ax in [ax1, ax2]:
        ax.coastlines()
        ax.gridlines()
    
    # Show the plot
    plt.show()
