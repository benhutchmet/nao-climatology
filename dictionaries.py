# Dictionary

# Path for the global psl data (.nc format)
ERA5_psl_global_path = "/home/users/benhutch/ERA5_psl/long-ERA5-full.nc"

# Path for the global tas data (.nc format)
ERA5_tas_wind_global_path = "/home/users/benhutch/ERA5/adaptor.mars.internal-1687448519.6842003-11056-8-3ea80a0a-4964-4995-bc42-7510a92e907b.nc"

# Lat range for the North Atlantic region
lat_range = slice(20, 70)     # Latitude range: 20°N to 70°N
lon_range = slice(-80, 20)    # Longitude range: 80°W to 20°E

# define the variable names
sfc_wind='si10'
tas='t2m'
psl='var151'

# define the azores and iceland grids
azores_grid = {
    'lon1': -28,
    'lon2': -20,
    'lat1': 36,
    'lat2': 40
}

iceland_grid = {
    'lon1': -25,
    'lon2': -16,
    'lat1': 63,
    'lat2': 70
}