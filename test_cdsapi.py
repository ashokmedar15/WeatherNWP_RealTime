import cdsapi

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['2m_temperature', 'surface_pressure'],  # Adjusted to include sp
        'year': '2025',
        'month': '01',
        'day': '01',
        'time': '00:00',
        'area': [24.5, 68.5, 20.5, 74.5],  # Gujarat
        'format': 'netcdf',
    },
    'test_era5_updated.nc'
)
print("CDS API test completed.")