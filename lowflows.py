import numpy as np
import pandas as pd
import s3fs
import xarray
import geopandas as gpd

bucket_uri = 's3://geoglows-v2-retrospective/retrospective.zarr'
region_name = 'us-west-2'
s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=region_name))
s3store = s3fs.S3Map(root=bucket_uri, s3=s3, check=False)

ds = xarray.open_zarr(s3store)
streams_122 =gpd.read_file(f'/Volumes/EB406_T7_2/source_streams/streams_122.gpkg')
print("GOT STREAMS")
percentile_95th_df = pd.read_csv("/Users/rachel1/Downloads/95th.csv")
streams_122_filtered = streams_122[~streams_122['LINKNO'].isin(percentile_95th_df['LINKNO'])]
i = 1
percentile_95th_dict = {}
for linkno in streams_122_filtered["LINKNO"]:
    df = ds['Qout'].sel(rivid=linkno).to_dataframe()
    df = df.reset_index().set_index('time').pivot(columns='rivid', values='Qout')
    df = df[(df.index > '1940-12-31')]
    data = df[linkno].sort_values().reset_index(drop=True)

    # Number of values
    n = len(data)

    # Weibull plotting positions
    data_ranked = data.reset_index()
    data_ranked.columns = ['rank', 'value']
    data_ranked['P'] = (data_ranked['rank'] + 1) / (n + 1)

    # Calculate the 95th percentile using interpolation
    percentile_95th = data_ranked[data_ranked['P'] >= 0.05].iloc[0]['value']
    percentile_95th_dict[linkno] = percentile_95th

    print(i)
    i = i + 1

# Convert the dictionary to a DataFrame
percentile_95th_df = pd.DataFrame.from_dict(percentile_95th_dict, orient='index', columns=['percentile_95th'])

# Reset the index to make 'LINKNO' a column
percentile_95th_df.reset_index(inplace=True)

# Rename the 'index' column to 'LINKNO'
percentile_95th_df.rename(columns={'index': 'LINKNO'}, inplace=True)

percentile_95th_df.to_csv("/Users/rachel1/Downloads/95th_others.csv")