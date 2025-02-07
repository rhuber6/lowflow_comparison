import s3fs
import xarray
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

rivids = [160279007,160196017, 160151592, 160722553, 160762267, 160772775, 160502764, 160501596, 160448969, 160064246]
#rivids = 160064246
bucket_uri = 's3://geoglows-v2-retrospective/retrospective.zarr'
region_name = 'us-west-2'
s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=region_name))
s3store = s3fs.S3Map(root=bucket_uri, s3=s3, check=False)

ds = xarray.open_zarr(s3store)

df = ds['Qout'].sel(rivid=rivids).to_dataframe()

df = df.reset_index().set_index('time').pivot(columns='rivid', values='Qout')
df = df[df.index > '1940-12-31']

monthly_avg = df.resample('ME').mean()

# Dictionary to store KS statistics and p-values for each distribution
results = {
    "Gamma": {"ks_statistics": [], "p_values": []},
    "Log-Pearson Type III": {"ks_statistics": [], "p_values": []},
    "Log-Normal": {"ks_statistics": [], "p_values": []},
    "Weibull": {"ks_statistics": [], "p_values": []},
    "Normal": {"ks_statistics": [], "p_values": []},
    "Gumbel": {"ks_statistics": [], "p_values": []}
}
monthly_avg['month'] = monthly_avg.index.month

# Iterate over each column in the DataFrame
"""
for column in monthly_avg.columns:
    # Drop NaN values in the column for fitting
    data = monthly_avg[column].dropna()

    # Gamma distribution
    shape, loc, scale = stats.gamma.fit(data, floc=0)
    ks_stat, p_val = stats.kstest(data, 'gamma', args=(shape, loc, scale))
    results["Gamma"]["ks_statistics"].append(ks_stat)
    results["Gamma"]["p_values"].append(p_val)

    # Log-Pearson Type III distribution (using log transformation and Pearson III)
    log_data = np.log(data)
    skew, loc, scale = stats.pearson3.fit(log_data)
    ks_stat, p_val = stats.kstest(log_data, 'pearson3', args=(skew, loc, scale))
    results["Log-Pearson Type III"]["ks_statistics"].append(ks_stat)
    results["Log-Pearson Type III"]["p_values"].append(p_val)

    # Log-Normal distribution
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    ks_stat, p_val = stats.kstest(data, 'lognorm', args=(shape, loc, scale))
    results["Log-Normal"]["ks_statistics"].append(ks_stat)
    results["Log-Normal"]["p_values"].append(p_val)

    # Weibull distribution
    shape, loc, scale = stats.weibull_min.fit(data, floc=0)
    ks_stat, p_val = stats.kstest(data, 'weibull_min', args=(shape, loc, scale))
    results["Weibull"]["ks_statistics"].append(ks_stat)
    results["Weibull"]["p_values"].append(p_val)

    # Normal distribution
    mean, std = stats.norm.fit(data)
    ks_stat, p_val = stats.kstest(data, 'norm', args=(mean, std))
    results["Normal"]["ks_statistics"].append(ks_stat)
    results["Normal"]["p_values"].append(p_val)
for dist_name, values in results.items():
    avg_ks_statistic = np.mean(values["ks_statistics"])
    avg_p_value = np.mean(values["p_values"])
    print(f"{dist_name} - Average KS Statistic: {avg_ks_statistic}, Average p-value: {avg_p_value}")


for month, group in monthly_avg.groupby('month'):
    data = group.iloc[:, 0]  # Assuming the values to fit are in the first column

    # Gamma distribution
    shape, loc, scale = stats.gamma.fit(data, floc=0)
    ks_stat, p_val = stats.kstest(data, 'gamma', args=(shape, loc, scale))
    results["Gamma"]["ks_statistics"].append((month, ks_stat))
    results["Gamma"]["p_values"].append((month, p_val))

    # Log-Pearson Type III distribution (log transformation + Pearson III)
    log_data = np.log(data[data > 0])  # Avoid issues with non-positive values
    skew, loc, scale = stats.pearson3.fit(log_data)
    ks_stat, p_val = stats.kstest(log_data, 'pearson3', args=(skew, loc, scale))
    results["Log-Pearson Type III"]["ks_statistics"].append((month, ks_stat))
    results["Log-Pearson Type III"]["p_values"].append((month, p_val))

    # Log-Normal distribution
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    ks_stat, p_val = stats.kstest(data, 'lognorm', args=(shape, loc, scale))
    results["Log-Normal"]["ks_statistics"].append((month, ks_stat))
    results["Log-Normal"]["p_values"].append((month, p_val))

    # Weibull distribution
    shape, loc, scale = stats.weibull_min.fit(data, floc=0)
    ks_stat, p_val = stats.kstest(data, 'weibull_min', args=(shape, loc, scale))
    results["Weibull"]["ks_statistics"].append((month, ks_stat))
    results["Weibull"]["p_values"].append((month, p_val))

    # Normal distribution
    mean, std = stats.norm.fit(data)
    ks_stat, p_val = stats.kstest(data, 'norm', args=(mean, std))
    results["Normal"]["ks_statistics"].append((month, ks_stat))
    results["Normal"]["p_values"].append((month, p_val))
# Calculate the average KS statistic and p-value for each distribution

average_results = {}

# Loop through each distribution in the results
for dist, metrics in results.items():
    # Extract KS statistics and p-values
    ks_statistics = [stat[1] for stat in metrics["ks_statistics"]]
    p_values = [pval[1] for pval in metrics["p_values"]]

    # Calculate averages
    avg_ks_stat = np.mean(ks_statistics)
    avg_p_val = np.mean(p_values)

    # Store the results
    average_results[dist] = {
        "average_ks_statistic": avg_ks_stat,
        "average_p_value": avg_p_val,
    }

# Display the results
for dist, averages in average_results.items():
    print(f"{dist}:")
    print(f"  Average KS Statistic: {averages['average_ks_statistic']:.4f}")
    print(f"  Average p-Value: {averages['average_p_value']:.4f}")
    """

dist_functions = {
    "Gamma": stats.gamma,
    "Log-Pearson Type III": stats.pearson3,
    "Log-Normal": stats.lognorm,
    "Weibull": stats.weibull_min,
    "Normal": stats.norm,
    "Gumbel": stats.gumbel_r,
}

# Process each month
for month, group in monthly_avg.groupby("month"):
    for column in group.columns.drop("month"):  # Exclude the 'month' column
        # Drop NaN values for the current column
        data = group[column].dropna()

        # Ensure there's enough data for fitting
        if len(data) > 1:
            # Gamma distribution
            shape, loc, scale = stats.gamma.fit(data, floc=0)
            ks_stat, p_val = stats.kstest(data, 'gamma', args=(shape, loc, scale))
            results["Gamma"]["ks_statistics"].append(ks_stat)
            results["Gamma"]["p_values"].append(p_val)

            # Log-Pearson Type III distribution (using log transformation and Pearson III)
            log_data = np.log(data)
            skew, loc, scale = stats.pearson3.fit(log_data)
            ks_stat, p_val = stats.kstest(log_data, 'pearson3', args=(skew, loc, scale))
            results["Log-Pearson Type III"]["ks_statistics"].append(ks_stat)
            results["Log-Pearson Type III"]["p_values"].append(p_val)

            # Log-Normal distribution
            shape, loc, scale = stats.lognorm.fit(data, floc=0)
            ks_stat, p_val = stats.kstest(data, 'lognorm', args=(shape, loc, scale))
            results["Log-Normal"]["ks_statistics"].append(ks_stat)
            results["Log-Normal"]["p_values"].append(p_val)

            # Weibull distribution
            shape, loc, scale = stats.weibull_min.fit(data, floc=0)
            ks_stat, p_val = stats.kstest(data, 'weibull_min', args=(shape, loc, scale))
            results["Weibull"]["ks_statistics"].append(ks_stat)
            results["Weibull"]["p_values"].append(p_val)

            # Normal distribution
            mean, std = stats.norm.fit(data)
            ks_stat, p_val = stats.kstest(data, 'norm', args=(mean, std))
            results["Normal"]["ks_statistics"].append(ks_stat)
            results["Normal"]["p_values"].append(p_val)

            # Gumbel distribution
            loc, scale = stats.gumbel_r.fit(data)
            ks_stat, p_val = stats.kstest(data, 'gumbel_r', args=(loc, scale))
            results["Gumbel"]["ks_statistics"].append(ks_stat)
            results["Gumbel"]["p_values"].append(p_val)
            # Initialize a figure for QQ plots
            fig, axes = plt.subplots(3, 2, figsize=(12, 18))
            axes = axes.flatten()

            for ax, (dist_name, dist_func) in zip(axes, dist_functions.items()):
                # Fit the data
                if dist_name == "Log-Pearson Type III":
                    params = dist_func.fit(np.log(data))
                    theoretical_quantiles = dist_func.ppf(
                        np.linspace(0.01, 0.99, len(data)), *params
                    )
                    observed_quantiles = np.sort(np.log(data))
                else:
                    params = dist_func.fit(data)
                    theoretical_quantiles = dist_func.ppf(
                        np.linspace(0.01, 0.99, len(data)), *params
                    )
                    observed_quantiles = np.sort(data)

                # Plot QQ
                ax.plot(
                    theoretical_quantiles,
                    observed_quantiles,
                    "o",
                    label=f"QQ Plot ({dist_name})",
                )
                ax.plot(
                    theoretical_quantiles,
                    theoretical_quantiles,
                    "r-",
                    label="1:1 Line",
                )
                ax.set_title(f"QQ Plot for {dist_name} ({month}-{column})")
                ax.set_xlabel("Theoretical Quantiles")
                ax.set_ylabel("Observed Quantiles")
                ax.legend()

            output_file = f"/Users/rachel1/Downloads/QQPlots/qq_plots_{month}_{column}.png"
            plt.tight_layout()
            plt.suptitle(f"QQ Plots for {column} in {month}", y=1.02)
            plt.savefig(output_file, dpi=600, bbox_inches="tight")
            plt.show()

# Calculate average KS-statistics and p-values for each distribution
average_results = {}

for dist, metrics in results.items():
    avg_ks_stat = np.mean(metrics["ks_statistics"])
    avg_p_val = np.mean(metrics["p_values"])
    average_results[dist] = {
        "average_ks_statistic": avg_ks_stat,
        "average_p_value": avg_p_val,
    }

# Display the average results
for dist, averages in average_results.items():
    print(f"{dist}:")
    print(f"  Average KS Statistic: {averages['average_ks_statistic']:.4f}")
    print(f"  Average p-Value: {averages['average_p_value']:.4f}")

