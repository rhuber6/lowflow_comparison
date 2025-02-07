import xarray
import geopandas as gpd
import numpy as np
import pandas as pd
import s3fs
import geoglows

# S3 setup
bucket_uri = 's3://geoglows-v2-retrospective/retrospective.zarr'
region_name = 'us-west-2'
s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=region_name))
s3store = s3fs.S3Map(root=bucket_uri, s3=s3, check=False)

# Load dataset
ds = xarray.open_zarr(s3store)

# Dictionary mapping linkno to their corresponding file paths
linkno_files = {
    160221792: "/Users/rachel1/Downloads/rachel nile/kenya_1GD03.csv",
    160266239: "/Users/rachel1/Downloads/rwanda_4326.csv",
    160212492: "/Users/rachel1/Downloads/rachel nile/grdc_1270900.csv",
    160168155: "/Users/rachel1/Downloads/rachel nile/grdc_1269200.csv",
    160213625: "/Users/rachel1/Downloads/rachel nile/grdc_1769050.csv",
    160184420: "/Users/rachel1/Downloads/rachel nile/grdc_1769100.csv",
    160191425: "/Users/rachel1/Downloads/rachel nile/grdc_1769200.csv",
    160128354: "/Users/rachel1/Downloads/rachel nile/grdc_1769150.csv",
    160504154: "/Users/rachel1/Downloads/rachel nile/grdc_1563680.csv",
    160528679: "/Users/rachel1/Downloads/rachel nile/grdc_1563700.csv",
    160622096: "/Users/rachel1/Downloads/rachel nile/grdc_1563900.csv",
    160590528: "/Users/rachel1/Downloads/rachel nile/grdc_1563600.csv",
    160536792: "/Users/rachel1/Downloads/rachel nile/grdc_1563450.csv",
    160596343: "/Users/rachel1/Downloads/rachel nile/grdc_1563500.csv",
    160608077: "/Users/rachel1/Downloads/rachel nile/grdc_1563550.csv",
    160553167: "/Users/rachel1/Downloads/rachel nile/grdc_1563050.csv",
}

results = []

# Loop through each linkno and open its corresponding CSV file
for linkno, file_path in linkno_files.items():
    print(linkno)
    df = ds['Qout'].sel(rivid=linkno).to_dataframe()
    df = df.reset_index().set_index('time').pivot(columns='rivid', values='Qout')
    df = df[(df.index > '1950-12-31')]
    df = df[df[linkno] >= 0]
    gauge = pd.read_csv(file_path)
    gauge = gauge[gauge['Streamflow (m3/s)'] >= 0]
    gauge['Datetime'] = pd.to_datetime(gauge['Datetime'], errors='coerce')
    gauge.set_index('Datetime', inplace=True)
    bias_correct = geoglows.bias.correct_historical(df, gauge)
    df.index = pd.to_datetime(df.index)

    merged_df = pd.merge(gauge, bias_correct, left_index=True, right_index=True, how='inner')

    # Calculate occurrences where 'Streamflow (m3/s)' is less than 1
    count_less_than_1_gauge = (merged_df['Streamflow (m3/s)'] < 1).sum()
    count_less_than_1_geoglows = (merged_df["Corrected Simulated Streamflow"] < 1).sum()

    # Filter out the rows where Streamflow is less than or equal to 0
    merged_df = merged_df[(merged_df['Streamflow (m3/s)'] > 0) & (merged_df["Corrected Simulated Streamflow"] > 0)]


    # Function to calculate the Weibull 95th percentile
    def weibull_95th_percentile(column):
        sorted_values = column.sort_values()
        n = len(sorted_values)
        weibull_positions = [(i + 1) / (n + 1) for i in range(n)]
        weibull_df = pd.DataFrame({'Values': sorted_values, 'Weibull Position': weibull_positions})
        value_95th_percentile = weibull_df.loc[weibull_df['Weibull Position'] >= 0.15, 'Values'].iloc[0]
        return value_95th_percentile


    # Calculate the 95th percentile for each column
    Q95_gauge = weibull_95th_percentile(merged_df['Streamflow (m3/s)'])
    Q95_geoglows = weibull_95th_percentile(merged_df["Corrected Simulated Streamflow"])

    # Count occurrences below 95th percentiles
    count_below_95th = \
    merged_df[(merged_df['Streamflow (m3/s)'] <= Q95_gauge) & (merged_df["Corrected Simulated Streamflow"] <= Q95_geoglows)].shape[0]

    # Calculate rolling averages and MAPE

    merged_df['GAUGE_7Day_Avg'] = merged_df['Streamflow (m3/s)'].rolling(window=7).mean()
    merged_df['GEOGLOWS_7Day_Avg'] = merged_df["Corrected Simulated Streamflow"].rolling(window=7).mean()

    count_7day_below_95th = merged_df[(merged_df['GAUGE_7Day_Avg'] < Q95_gauge) & (merged_df['GEOGLOWS_7Day_Avg'] < Q95_geoglows)].shape[0]

    merged_df['MAPE_Streamflow_vs_GEOGLOWS'] = abs(
        (merged_df['Streamflow (m3/s)'] - merged_df["Corrected Simulated Streamflow"]) / merged_df['Streamflow (m3/s)']) * 100
    merged_df['MAPE_GAUGE_vs_GEOGLOWS'] = abs(
        (merged_df['GAUGE_7Day_Avg'] - merged_df['GEOGLOWS_7Day_Avg']) / merged_df['GAUGE_7Day_Avg']) * 100

    MAPE_Streamflow_vs_GEOGLOWS = merged_df['MAPE_Streamflow_vs_GEOGLOWS'].mean()
    MAPE_GAUGE_vs_GEOGLOWS = merged_df['MAPE_GAUGE_vs_GEOGLOWS'].mean()

    # Filtered data based on conditions for MAPE calculation
    filtered_df = merged_df[(merged_df['GAUGE_7Day_Avg'] < Q95_gauge) | (merged_df['GEOGLOWS_7Day_Avg'] < Q95_geoglows)]

    # Avoid SettingWithCopyWarning
    filtered_df = filtered_df.copy()

    filtered_df['MAPE_Streamflow_vs_160266239'] = abs(
        (filtered_df['Streamflow (m3/s)'] - filtered_df["Corrected Simulated Streamflow"]) / filtered_df['Streamflow (m3/s)']) * 100
    filtered_df['MAPE_GAUGE_vs_GEOGLOWS'] = abs(
        (filtered_df['GAUGE_7Day_Avg'] - filtered_df['GEOGLOWS_7Day_Avg']) / filtered_df['GAUGE_7Day_Avg']) * 100

    MAPE_Streamflow_vs_160266239_filtered = filtered_df['MAPE_Streamflow_vs_160266239'].mean()
    MAPE_GAUGE_vs_GEOGLOWS_filtered = filtered_df['MAPE_GAUGE_vs_GEOGLOWS'].mean()

    # Append the results for this linkno to the list
    results.append({
        'linkno': linkno,
        'count_less_than_1_gauge': count_less_than_1_gauge,
        'count_less_than_1_geoglows': count_less_than_1_geoglows,
        'Q95_gauge': Q95_gauge,
        'Q95_geoglows': Q95_geoglows,
        'count_below_95th': count_below_95th,
        'count_7day_below_95th': count_7day_below_95th,
        'MAPE': MAPE_Streamflow_vs_GEOGLOWS,
        'MAPE_rollling_avg': MAPE_GAUGE_vs_GEOGLOWS,
        'MAPE_filtered': MAPE_Streamflow_vs_160266239_filtered,
        'MAPE_rolling_avg_filtered': MAPE_GAUGE_vs_GEOGLOWS_filtered
    })

results_df = pd.DataFrame(results)

results_df.to_csv("/Users/rachel1/Downloads/bias_corrected_lowflow_results_Q85.csv")
