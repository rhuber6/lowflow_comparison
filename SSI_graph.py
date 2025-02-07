import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import calendar
import s3fs
import xarray

rivids = [160279007, 160196017, 160151592, 160722553, 160762267, 160772775, 160502764, 160501596, 160448969, 160064246]

bucket_uri = 's3://geoglows-v2-retrospective/retrospective.zarr'
region_name = 'us-west-2'
s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=region_name))
s3store = s3fs.S3Map(root=bucket_uri, s3=s3, check=False)

ds = xarray.open_zarr(s3store)

df = ds['Qout'].sel(rivid=rivids).to_dataframe()

df = df.reset_index().set_index('time').pivot(columns='rivid', values='Qout')

df = df[df.index > '1940-12-31']

retrospective_data = df


def get_SSI(df):
    result_df = pd.DataFrame()
    for month in range(1, 13):  # Months are from 1 to 12
        monthly_average = df.resample('M').mean()
        filtered_df = monthly_average[monthly_average.index.month == month].copy()
        mean = filtered_df.iloc[:, 0].mean()
        std_dev = filtered_df.iloc[:, 0].std()
        filtered_df['cumulative_probability'] = filtered_df.iloc[:, 0].apply(
            lambda x: 1 - stats.norm.cdf(x, mean, std_dev))
        filtered_df['probability_less_than_0.5'] = filtered_df['cumulative_probability'] < 0.5
        filtered_df['p'] = filtered_df['cumulative_probability']
        filtered_df.loc[filtered_df['cumulative_probability'] > 0.5, 'p'] = 1 - filtered_df['cumulative_probability']
        filtered_df['W'] = (-2 * np.log(filtered_df['p'])) ** 0.5
        C0 = 2.515517
        C1 = 0.802853
        C2 = 0.010328
        d1 = 1.432788
        d2 = 0.001308
        d3 = 0.001308
        filtered_df['SSI'] = filtered_df['W'] - (C0 + C1 * filtered_df['W'] + C2 * filtered_df['W'] ** 2) / (
                1 + d1 * filtered_df['W'] + d2 * filtered_df['W'] ** 2 + d3 * filtered_df['W'] ** 3)
        filtered_df.loc[filtered_df['probability_less_than_0.5'] == False, 'SSI'] *= -1
        # month_df = pd.DataFrame({'Month': [month] * len(filtered_df), 'SSI': filtered_df['SSI'].values})
        result_df = pd.concat([result_df, filtered_df])
    return result_df


# Assuming df is your DataFrame with a datetime index
ssi_result_normal = get_SSI(retrospective_data)
ssi_result_sorted_normal = ssi_result_normal.sort_index()
ssi_result_sorted_normal = ssi_result_sorted_normal['2010':]


def get_SSI_weibull(df):
    result_df = pd.DataFrame()

    for month in range(1, 13):  # Months are from 1 to 12
        monthly_average = df.resample('M').mean()
        filtered_df = monthly_average[monthly_average.index.month == month].copy()

        # Sort the data and calculate ranks
        filtered_df = filtered_df.sort_values(by=filtered_df.columns[0])
        n = len(filtered_df)
        filtered_df['rank'] = np.arange(1, n + 1)

        # Calculate Weibull cumulative probability
        filtered_df['cumulative_probability'] = filtered_df['rank'] / (n + 1)

        filtered_df['probability_less_than_0.5'] = filtered_df['cumulative_probability'] < 0.5
        filtered_df['p'] = filtered_df['cumulative_probability']
        filtered_df.loc[filtered_df['cumulative_probability'] < 0.5, 'p'] = 1 - filtered_df['cumulative_probability']

        # Calculate W using the transformed p
        filtered_df['W'] = (-2 * np.log(filtered_df['p'])) ** 0.5

        # Constants for the calculation of SSI
        C0 = 2.515517
        C1 = 0.802853
        C2 = 0.010328
        d1 = 1.432788
        d2 = 0.001308
        d3 = 0.001308

        # Calculate SSI using the W values
        filtered_df['SSI'] = filtered_df['W'] - (C0 + C1 * filtered_df['W'] + C2 * filtered_df['W'] ** 2) / (
                1 + d1 * filtered_df['W'] + d2 * filtered_df['W'] ** 2 + d3 * filtered_df['W'] ** 3)

        filtered_df.loc[filtered_df['probability_less_than_0.5'] == False, 'SSI'] *= -1

        # Combine results
        result_df = pd.concat([result_df, filtered_df])

    return result_df


# Assuming df is your DataFrame with a datetime index
ssi_result_weibull = get_SSI_weibull(retrospective_data)
ssi_result_sorted_weibull = ssi_result_weibull.sort_index()
ssi_result_sorted_weibull = ssi_result_sorted_weibull['2010':]


def get_SSI_gamma(df):
    result_df = pd.DataFrame()

    for month in range(1, 13):  # Months are from 1 to 12
        monthly_average = df.resample('M').mean()
        filtered_df = monthly_average[monthly_average.index.month == month].copy()

        # Estimate parameters for the gamma distribution
        shape, loc, scale = stats.gamma.fit(filtered_df.iloc[:, 0], floc=0)  # Fix location to 0

        # Calculate cumulative probability using the gamma CDF
        filtered_df['cumulative_probability'] = filtered_df.iloc[:, 0].apply(
            lambda x: 1 - stats.gamma.cdf(x, shape, loc, scale)
        )

        filtered_df['probability_less_than_0.5'] = filtered_df['cumulative_probability'] < 0.5
        filtered_df['p'] = filtered_df['cumulative_probability']
        filtered_df.loc[filtered_df['cumulative_probability'] > 0.5, 'p'] = 1 - filtered_df['cumulative_probability']

        # Calculate W using the transformed p
        filtered_df['W'] = (-2 * np.log(filtered_df['p'])) ** 0.5

        # Constants for the calculation of SSI
        C0 = 2.515517
        C1 = 0.802853
        C2 = 0.010328
        d1 = 1.432788
        d2 = 0.001308
        d3 = 0.001308

        # Calculate SSI using the W values
        filtered_df['SSI'] = filtered_df['W'] - (C0 + C1 * filtered_df['W'] + C2 * filtered_df['W'] ** 2) / (
                1 + d1 * filtered_df['W'] + d2 * filtered_df['W'] ** 2 + d3 * filtered_df['W'] ** 3)

        filtered_df.loc[filtered_df['probability_less_than_0.5'] == False, 'SSI'] *= -1

        # Combine results
        result_df = pd.concat([result_df, filtered_df])

    return result_df


def get_SSI_lognormal(df):
    result_df = pd.DataFrame()

    for month in range(1, 13):  # Months are from 1 to 12
        monthly_average = df.resample('M').mean()
        filtered_df = monthly_average[monthly_average.index.month == month].copy()

        # Apply log transformation to positive values only
        log_data = np.log(filtered_df.iloc[:, 0][filtered_df.iloc[:, 0] > 0])

        # Estimate parameters for the lognormal distribution
        shape, loc, scale = stats.lognorm.fit(np.exp(log_data), floc=0)  # Using the original (positive) data

        # Calculate cumulative probability using the lognormal CDF
        filtered_df['cumulative_probability'] = filtered_df.iloc[:, 0].apply(
            lambda x: 1 - stats.lognorm.cdf(x, shape, loc, scale) if x > 0 else np.nan
        )

        filtered_df['probability_less_than_0.5'] = filtered_df['cumulative_probability'] < 0.5
        filtered_df['p'] = filtered_df['cumulative_probability']
        filtered_df.loc[filtered_df['cumulative_probability'] > 0.5, 'p'] = 1 - filtered_df['cumulative_probability']

        # Calculate W using the transformed p
        filtered_df['W'] = (-2 * np.log(filtered_df['p'])) ** 0.5

        # Constants for the calculation of SSI
        C0 = 2.515517
        C1 = 0.802853
        C2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308

        # Calculate SSI using the W values
        filtered_df['SSI'] = filtered_df['W'] - (C0 + C1 * filtered_df['W'] + C2 * filtered_df['W'] ** 2) / (
                1 + d1 * filtered_df['W'] + d2 * filtered_df['W'] ** 2 + d3 * filtered_df['W'] ** 3)

        filtered_df.loc[filtered_df['probability_less_than_0.5'] == False, 'SSI'] *= -1

        # Combine results
        result_df = pd.concat([result_df, filtered_df])

    return result_df


# Assuming df is your DataFrame with a datetime index
ssi_result_lognormal = get_SSI_lognormal(df)

ssi_result_sorted_lognormal = ssi_result_lognormal.sort_index()
ssi_result_sorted_lognormal = ssi_result_sorted_lognormal['2010':]
# Assuming df is your DataFrame with a datetime index
ssi_result_gamma = get_SSI_gamma(retrospective_data)

ssi_result_sorted_gamma = ssi_result_gamma.sort_index()
ssi_result_sorted_gamma = ssi_result_sorted_gamma['2010':]


def get_SSI_log_pearson3(df):
    result_df = pd.DataFrame()

    for month in range(1, 13):  # Months are from 1 to 12
        monthly_average = df.resample('M').mean()
        filtered_df = monthly_average[monthly_average.index.month == month].copy()

        # Apply logarithmic transformation to ensure no non-positive values
        log_data = np.log(filtered_df.iloc[:, 0][filtered_df.iloc[:, 0] > 0])

        # Estimate parameters for the Pearson Type III distribution
        skew, loc, scale = stats.pearson3.fit(log_data)

        # Calculate cumulative probability using the Pearson Type III CDF
        filtered_df['cumulative_probability'] = filtered_df.iloc[:, 0].apply(
            lambda x: 1 - stats.pearson3.cdf(np.log(x), skew, loc, scale) if x > 0 else np.nan
        )

        filtered_df['probability_less_than_0.5'] = filtered_df['cumulative_probability'] < 0.5
        filtered_df['p'] = filtered_df['cumulative_probability']
        filtered_df.loc[filtered_df['cumulative_probability'] > 0.5, 'p'] = 1 - filtered_df['cumulative_probability']

        # Calculate W using the transformed p
        filtered_df['W'] = (-2 * np.log(filtered_df['p'])) ** 0.5

        # Constants for the calculation of SSI
        C0 = 2.515517
        C1 = 0.802853
        C2 = 0.010328
        d1 = 1.432788
        d2 = 0.001308
        d3 = 0.001308

        # Calculate SSI using the W values
        filtered_df['SSI'] = filtered_df['W'] - (C0 + C1 * filtered_df['W'] + C2 * filtered_df['W'] ** 2) / (
                1 + d1 * filtered_df['W'] + d2 * filtered_df['W'] ** 2 + d3 * filtered_df['W'] ** 3)

        filtered_df.loc[filtered_df['probability_less_than_0.5'] == False, 'SSI'] *= -1

        # Combine results
        result_df = pd.concat([result_df, filtered_df])

    return result_df


# Assuming df is your DataFrame with a datetime index
ssi_result_log_pearson3 = get_SSI_log_pearson3(retrospective_data)

ssi_result_sorted_log_pearson3 = ssi_result_log_pearson3.sort_index()
ssi_result_sorted_log_pearson3 = ssi_result_sorted_log_pearson3['2010':]


def plot_ssi_results(ssi_normal, ssi_lognormal, ssi_gamma, ssi_pearson, ssi_weibull):
    plt.figure(figsize=(14, 7))

    # Plot SSI from normal distribution
    plt.plot(ssi_normal.index, ssi_normal['SSI'], label='Normal Distribution SSI', color='blue', alpha=0.7)

    # Plot SSI from lognormal distribution
    plt.plot(ssi_lognormal.index, ssi_lognormal['SSI'], label='Lognormal Distribution SSI', color='red', alpha=0.7)

    # Plot SSI from gamma distribution
    plt.plot(ssi_gamma.index, ssi_gamma['SSI'], label='Gamma Distribution SSI', color='green', alpha=0.7)

    # Plot SSI from Pearson Type III distribution
    plt.plot(ssi_pearson.index, ssi_pearson['SSI'], label='Pearson Type III Distribution SSI', color='purple',
             alpha=0.7)

    # Plot SSI from Weibull distribution
    plt.plot(ssi_weibull.index, ssi_weibull['SSI'], label='Weibull Distribution SSI', color='orange', alpha=0.7)

    plt.title('SSI Values for Different Distributions')
    plt.xlabel('Date')
    plt.ylabel('SSI')
    plt.legend()
    plt.grid(True)
    # plt.savefig("/Users/rachel1/Downloads/high_resolution_for_thesis/SSI_comparison.png", dpi =500)
    plt.show()


# Assuming you have already calculated ssi_result_normal and ssi_result_log_normal
plot_ssi_results(ssi_result_sorted_normal, ssi_result_sorted_lognormal, ssi_result_sorted_gamma,
                 ssi_result_sorted_log_pearson3, ssi_result_sorted_weibull)
plt.figure(figsize=(10, 6))
plt.plot(ssi_result_sorted_log_pearson3.index, ssi_result_sorted_log_pearson3['SSI'], color='blue', marker='o', linestyle='-', markersize=3)
plt.title('SSI Monthly Values Over Time')
plt.xlabel('Date')
plt.ylabel('SSI')
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/rachel1/Downloads/high_resolution_for_thesis/ssi_plot.png", dpi=600)
plt.show()