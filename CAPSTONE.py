import xarray
import geopandas as gpd
import numpy as np
import pandas as pd
import s3fs
import geoglows

bucket_uri = 's3://geoglows-v2-retrospective/retrospective.zarr'
region_name = 'us-west-2'
s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=region_name))
s3store = s3fs.S3Map(root=bucket_uri, s3=s3, check=False)

ds = xarray.open_zarr(s3store)
linkno = 160266239
df = ds['Qout'].sel(rivid=linkno).to_dataframe()
df = df.reset_index().set_index('time').pivot(columns='rivid', values='Qout')
df = df[(df.index > '1950-12-31')]
df = df[df[linkno] >= 0]
df.reset_index(inplace=True)
print(df)
print("SUCCESS!")

linkno_files = {
    160266239: ("/Users/rachel1/Downloads/rwanda_4326.csv"),
    160205470: ('/Users/rachel1/Downloads/rachel nile/fix/221001.csv'),
    160233531: ('/Users/rachel1/Downloads/rachel nile/fix/270001.csv'),

}