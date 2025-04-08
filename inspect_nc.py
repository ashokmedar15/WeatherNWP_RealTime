from netCDF4 import Dataset

nc_file = Dataset('test_era5_updated.nc', 'r')
print("Variables:", nc_file.variables.keys())
for var in nc_file.variables:
    print(f"{var}: {nc_file.variables[var].shape}")
print("Dimensions:", nc_file.dimensions.keys())
for dim in nc_file.dimensions:
    print(f"{dim}: {len(nc_file.dimensions[dim])}")
nc_file.close()