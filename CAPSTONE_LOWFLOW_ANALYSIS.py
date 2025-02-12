import s3fs
import xarray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

bucket_uri = 's3://geoglows-v2-retrospective/retrospective.zarr'
region_name = 'us-west-2'
s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=region_name))
s3store = s3fs.S3Map(root=bucket_uri, s3=s3, check=False)

ds = xarray.open_zarr(s3store)

results = []
# Replace linkno_files with the dictionary of linkno and file paths
linkno_files = {
    670031759: ("/Users/rachel1/Downloads/87-01-01_Q.csv"),

}
# Code to make sure that we are correctly reading in the observed data

stdStart = 1991
stdEnd = 2020


def process_groupby(flowdata):
    flowdata.columns = ['date', 'flow']
    flowdata['date'] = pd.to_datetime(flowdata['date'])  # Ensure the 'date' column is in datetime format

    # Check dates are sequential
    diff = pd.date_range(start=flowdata['date'].min(), end=flowdata['date'].max()).difference(flowdata['date'])
    if len(diff) > 0:
        flowdata.set_index('date', inplace=True)
        for md in diff:
            flowdata.loc[md, 'flow'] = pd.NA
        flowdata.reset_index(inplace=True)

    # Add month and year columns
    flowdata['month'] = flowdata['date'].dt.month
    flowdata['year'] = flowdata['date'].dt.year

    # Print data completeness
    print(f"There are {flowdata['year'].max() - flowdata['year'].min()} years of data in this file.")
    print(
        f"There are {sum(flowdata['flow'].isnull())} missing data points, "
        f"which is {np.round(sum(flowdata['flow'].isnull()) / len(flowdata) * 100, 4)}% of the total data"
    )

    # Group by month and year, calculate percentage completeness and mean flows
    groupBy = (flowdata.groupby(['month', 'year']).count()['flow'] /
               flowdata.groupby(['month', 'year']).count()['date']) * 100

    groupBy = pd.DataFrame(groupBy, columns=['monthly%'])
    groupBy['mean_flow'] = flowdata.groupby(['month', 'year'])['flow'].mean()

    # set the mean flow to NAN if there is less than 50 % data
    groupBy.loc[groupBy['monthly%'] < 50, 'mean_flow'] = pd.NA
    groupBy.reset_index(inplace=True)
    groupBy = groupBy.dropna()

    return groupBy


def calculate_lta_and_flow_categories(groupBy):
    # Calculate long-term average (LTA)
    LTA = groupBy[(groupBy['year'] >= stdStart) & (groupBy['year'] <= stdEnd)].groupby(['month'])['mean_flow'].mean()
    if LTA.empty:
        return None

    # Divide each month by the LTA and calculate rank percentiles
    for i in range(1, 13):
        groupBy.loc[groupBy['month'] == i, 'percentile_flow'] = (
                groupBy['mean_flow'][groupBy['month'] == i] / LTA[i] * 100
        )
        groupBy.loc[groupBy['month'] == i, 'weibell_rank'] = groupBy.loc[
                                                                 groupBy['month'] == i, 'percentile_flow'
                                                             ].rank(na_option='keep') / (groupBy.loc[groupBy[
                                                                                                         'month'] == i, 'percentile_flow'].count() + 1)

    # Assign flow categories
    def flow_status(weibell_rank):
        if weibell_rank <= 0.13:
            return 1
        elif weibell_rank <= 0.28:
            return 2
        elif weibell_rank <= 0.71999:
            return 3
        elif weibell_rank <= 0.86999:
            return 4
        else:
            return 5

    groupBy['flowcat'] = groupBy['weibell_rank'].apply(flow_status)

    # Format the output
    groupBy['date'] = pd.to_datetime(groupBy[['year', 'month']].assign(DAY=1))
    groupBy['date'] = groupBy['date'].dt.strftime('%Y-%m-%d')
    groupBy['flowcat'] = groupBy['flowcat'].astype('Int64')

    return groupBy.sort_values(['year', 'month']).filter(['date', 'flowcat'])


results = pd.DataFrame()
heatmaps = []
titles = []

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for linkno, (file_path) in linkno_files.items():
    print(linkno)
    df = ds['Qout'].sel(rivid=linkno).to_dataframe()
    df = df.reset_index().set_index('time').pivot(columns='rivid', values='Qout')
    df = df[(df.index > '1950-12-31')]
    df = df[df[linkno] >= 0]
    df.reset_index(inplace=True)

    # This needs to be fixed to read in the observed data correctly
    # start_row = find_start_row(file_path)
    gauge = pd.read_csv(
        file_path,
        parse_dates=[0]
    )
    gauge.columns = ['Datetime', 'Streamflow (m3/s)']
    gauge = gauge[gauge['Streamflow (m3/s)'] >= 0]
    # If no specific format is known and you need to handle two-digit years:
    gauge['Datetime'] = pd.to_datetime(
        gauge['Datetime'], errors='coerce', yearfirst=True
    )

    # Explicitly correct any misinterpreted years
    # Assuming dates later than the current year are incorrect:
    current_year = pd.Timestamp.now().year
    gauge['Datetime'] = gauge['Datetime'].apply(
        lambda x: x.replace(year=x.year - 100) if x.year > current_year else x
    )
    # gauge['Datetime'] = pd.to_datetime(gauge['Datetime'], errors='coerce')
    gauge.set_index('Datetime', inplace=True)
    gauge.reset_index(inplace=True)

    unique_years = gauge['Datetime'].dt.year.nunique()
    if unique_years < 3:
        print(f"Skipping linkno {linkno} due to insufficient data ({unique_years} years)")
        continue

    geoglows_monthly = process_groupby(df)

    # print("FINISHED GEOGLOWS")
    gauge_monthly = process_groupby(gauge)

    geoglows_filtered = geoglows_monthly.merge(
        gauge_monthly[['year', 'month']],  # Select only the columns needed for matching
        on=['year', 'month'],
        how='inner'  # Keep only rows that match
    )

    geoglows_sos = calculate_lta_and_flow_categories(geoglows_filtered)

    gauge_sos = calculate_lta_and_flow_categories(gauge_monthly)

    if gauge_sos is not None:
        # Rename the value columns while preserving the 'time' column
        gauge_sos.rename(columns={gauge_sos.columns[1]: f"{linkno}_gauge"}, inplace=True)
        geoglows_sos.rename(columns={geoglows_sos.columns[1]: f"{linkno}_geoglows"}, inplace=True)

        # Combine the DataFrames along columns, aligning by 'time'
        combined_df = pd.merge(
            gauge_sos, geoglows_sos, on=gauge_sos.columns[0], how="outer"
        )

        if results.empty:
            # Initialize the results DataFrame with the first pair of data
            results = pd.merge(
                gauge_sos, geoglows_sos, on=gauge_sos.columns[0], how="outer"
            )
        else:
            # Merge new results into the existing DataFrame
            results = pd.merge(
                results, gauge_sos, on=gauge_sos.columns[0], how="outer"
            )
            results = pd.merge(
                results, geoglows_sos, on=geoglows_sos.columns[0], how="outer"
            )
        # Ensure 'date' column is in datetime format
        combined_df["date"] = pd.to_datetime(combined_df["date"])
        combined_df["abs_diff"] = abs(combined_df[f"{linkno}_gauge"] - combined_df[f"{linkno}_geoglows"])
        monthly_abs_diff = []
        print(f"NUMBER IF MONTHS{combined_df.shape[0]}")

        # Loop through each month
        for month in range(1, 13):
            # Filter data for the current month
            month_data = combined_df[combined_df["date"].dt.month == month]
            # Calculate the sum of 'abs_diff' for the current month
            total_abs_diff = month_data["abs_diff"].mean()

            # Append the result to the list with the associated month
            monthly_abs_diff.append({"Month": month, "Total_Abs_Diff": total_abs_diff})

            # Skip if there's no data for the month
            if month_data.empty:
                continue

            # Plot the data for the current month
            plt.figure(figsize=(10, 6))

            # Create bar plots for Gauge and GeoGLOWS data
            width = 0.4  # Width of each bar
            x = month_data["date"]

            # Plot the data for the current month
            plt.figure(figsize=(10, 6))

            # Create the scatter points
            # Define colors based on whether gauge and geoglows values match
            colors = ['green' if gauge_val == geoglows_val else 'blue' for gauge_val, geoglows_val in
                      zip(month_data[f"{linkno}_gauge"], month_data[f"{linkno}_geoglows"])]

            # Create scatter points for gauge
            plt.scatter(month_data["date"], month_data[f"{linkno}_gauge"], color=colors, edgecolor='black', s=50,
                        marker='o', label='Gauge')

            # Create scatter points for geoglows (excluding green because it's already plotted)
            plt.scatter(month_data["date"], month_data[f"{linkno}_geoglows"],
                        color=['orange' if c != 'green' else 'green' for c in colors],
                        edgecolor='black', s=50, marker='o', label='GHM HydroSOS')

            # Add vertical lines to connect points from the same date
            for date, gauge_val, geoglows_val in zip(month_data["date"], month_data[f"{linkno}_gauge"],
                                                     month_data[f"{linkno}_geoglows"]):
                plt.vlines(x=date, ymin=gauge_val, ymax=geoglows_val, colors='gray', linestyles='dashed')

            # Add titles, labels, and legend
            if not month_data.empty:
                month_name = month_data["date"].dt.strftime('%B').iloc[0]
            else:
                month_name = "Unknown"

            plt.title(f'Comparison of Gauge and GHM HydroSOS Categories of {linkno} for {month_name}', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Values', fontsize=12)

            # Update y-axis labels
            y_labels = {1: "Low Flow", 2: "Below Normal", 3: "Normal", 4: "Above Normal", 5: "High Flow"}
            plt.yticks(ticks=list(y_labels.keys()), labels=list(y_labels.values()))

            # **Fix the Legend:**
            blue_marker = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='Gauge')
            orange_marker = mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=8,
                                          label='GHM HydroSOS')
            green_marker = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=8,
                                         label='Agreement (Gauge & GHM)')
            plt.legend(handles=[blue_marker, orange_marker, green_marker], title="Data Source")

            plt.grid(True)

            # Ensure the directory exists before saving
            #FIX THIS FILE PATH
            save_path = f'/Users/rachel1/Downloads/rachel nile/heatmaps/AAAAAcomparison_{linkno}_{month_name}.png'

            plt.savefig(save_path, dpi=500, bbox_inches='tight')

            # Show the plot
            plt.show()
