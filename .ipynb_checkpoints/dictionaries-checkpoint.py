# Dictionary

# Path for the global psl data (.nc format)
ERA5_psl_global_path = "/home/users/benhutch/ERA5_psl/long-ERA5-full.nc"

# Path for the global tas data (.nc format)
ERA5_tas_wind_global_path = "/home/users/benhutch/ERA5/adaptor.mars.internal-1687448519.6842003-11056-8-3ea80a0a-4964-4995-bc42-7510a92e907b.nc"

# define the variable names
sfc_wind='si10'
tas='t2m'
psl='var151'

# define the azores and iceland grids
azores_grid = {
    'lon1': 332,
    'lon2': 340,
    'lat1': 40,
    'lat2': 36
}

iceland_grid = {
    'lon1': 335,
    'lon2': 344,
    'lat1': 70,
    'lat2': 63
}

# define the north atlantic grid
north_atlantic_grid = {
    'lon1': 300,
    'lon2': 20,
    'lat1': 80,
    'lat2': 30
}

north_atlantic_grid_roll = {
    'lon1': 0,
    'lon2': 10,
    'lat1': 80,
    'lat2': 30
}



# # define the azores and iceland grids
# azores_grid = {
#     'lon1': -28,
#     'lon2': -20,
#     'lat1': 36,
#     'lat2': 40
# }

# iceland_grid = {
#     'lon1': -25,
#     'lon2': -16,
#     'lat1': 63,
#     'lat2': 70
# }