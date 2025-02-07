import xarray
import geopandas as gpd
import numpy as np
import pandas as pd
import s3fs
import geoglows
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

bucket_uri = 's3://geoglows-v2-retrospective/retrospective.zarr'
region_name = 'us-west-2'
s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=region_name))
s3store = s3fs.S3Map(root=bucket_uri, s3=s3, check=False)

ds = xarray.open_zarr(s3store)

results = []

linkno_files = {
    #160221792: "/Users/rachel1/Downloads/rachel nile/kenya_1GD03.csv",
    160266239: ("/Users/rachel1/Downloads/rwanda_4326.csv", 0),
    # 160194980: '/Users/rachel1/Downloads/rachel nile/fix/70012.csv',
    #130613583: '/Users/rachel1/Downloads/rachel nile/fix/196501_2.csv',
    160205470: ('/Users/rachel1/Downloads/rachel nile/fix/221001.csv', 1),
    # 160266239: '/Users/rachel1/Downloads/rachel nile/fix/259501.csv',
    160233531: ('/Users/rachel1/Downloads/rachel nile/fix/270001.csv', 2),
    # 160200814: '/Users/rachel1/Downloads/rachel nile/fix/294701.csv',
    # 160212524: '/Users/rachel1/Downloads/rachel nile/fix/298001.csv',
    # 160217207: '/Users/rachel1/Downloads/rachel nile/fix/SW5.csv',
    # 160157611: '/Users/rachel1/Downloads/rachel nile/fix/SW27.csv',
    # 160157611: '/Users/rachel1/Downloads/rachel nile/fix/SW28.csv',
    # 160157611: '/Users/rachel1/Downloads/rachel nile/fix/SW29.csv',
    # 160157611: '/Users/rachel1/Downloads/rachel nile/fix/SW30.csv',
    # 160161112: '/Users/rachel1/Downloads/rachel nile/fix/282001.csv'
    # 160212492: "/Users/rachel1/Downloads/rachel nile/grdc_1270900.csv",
    # 160168155: "/Users/rachel1/Downloads/rachel nile/grdc_1269200.csv",
    # 160213625: "/Users/rachel1/Downloads/rachel nile/grdc_1769050.csv",
    # 160184420: "/Users/rachel1/Downloads/rachel nile/grdc_1769100.csv",
    # 160191425: "/Users/rachel1/Downloads/rachel nile/grdc_1769200.csv",
    # 160128354: "/Users/rachel1/Downloads/rachel nile/grdc_1769150.csv",
    # 160504154: "/Users/rachel1/Downloads/rachel nile/grdc_1563680.csv",
    # 160528679: "/Users/rachel1/Downloads/rachel nile/grdc_1563700.csv",
    # 160622096: "/Users/rachel1/Downloads/rachel nile/grdc_1563900.csv",
    # 160590528: "/Users/rachel1/Downloads/rachel nile/grdc_1563600.csv",
    # 160536792: "/Users/rachel1/Downloads/rachel nile/grdc_1563450.csv",
    # 160596343: "/Users/rachel1/Downloads/rachel nile/grdc_1563500.csv",
    # 160608077: "/Users/rachel1/Downloads/rachel nile/grdc_1563550.csv",
    # 160553167: "/Users/rachel1/Downloads/rachel nile/grdc_1563050.csv",

}


def find_start_row(file_path, column_index=1):
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            # Split the line by comma and check the second column
            columns = line.strip().split(',')
            if len(columns) > column_index:
                try:
                    float(columns[column_index])  # Check if it's a number
                    return i  # Return the row number
                except ValueError:
                    continue
    return 0



import pandas as pd
import numpy as np

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

stdStart = 1991
stdEnd = 2020

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
        ].rank(na_option='keep') / (groupBy.loc[groupBy['month'] == i, 'percentile_flow'].count() + 1)

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

for linkno, (file_path, i) in linkno_files.items():
    print(linkno)
    df = ds['Qout'].sel(rivid=linkno).to_dataframe()
    df = df.reset_index().set_index('time').pivot(columns='rivid', values='Qout')
    df = df[(df.index > '1950-12-31')]
    df = df[df[linkno] >= 0]
    df.reset_index(inplace=True)
    start_row = find_start_row(file_path)
    gauge = pd.read_csv(
        file_path,
        skiprows=start_row,
        header=None,  # Do not use the first row as column names
        parse_dates=[0]  # Parse the first column as datetime
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
        how='inner'                       # Keep only rows that match
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

            # Create bar positions
            x_indices = range(len(x))  # Numeric indices for the x-axis
            #plt.bar([xi - width / 2 for xi in x_indices], month_data[f"{linkno}_gauge"], width=width, label='Gauge',
             #       color='blue')
            #plt.bar([xi + width / 2 for xi in x_indices], month_data[f"{linkno}_geoglows"], width=width,
             #       label='GeoGLOWS', color='orange')

            # Add titles, labels, and legend
           # month_name = month_data["date"].dt.strftime('%B').iloc[0]  # Get month name
            #plt.title(f'Comparison of {linkno}_Gauge and {linkno}_GeoGLOWS for {month_name}', fontsize=14)
            #plt.xlabel('Date', fontsize=12)
            #plt.ylabel('HydroSOS Category', fontsize=12)
            #plt.xticks(ticks=x_indices, labels=x.dt.strftime('%d-%b'), rotation=45)  # Use formatted dates for x-axis
            #plt.legend()
            #plt.grid(True, axis='y')  # Optional: Grid only on the y-axis for a bar graph
            #plt.tight_layout()  # Adjust layout to prevent overlap
            # plt.show()

            # Plot the data for the current month
            plt.figure(figsize=(10, 6))

            # Plot lines
            plt.plot(month_data["date"], month_data[f"{linkno}_gauge"], label='Gauge', color='blue', linewidth=2)
            plt.plot(month_data["date"], month_data[f"{linkno}_geoglows"], label='GeoGLOWS', color='orange',
                     linewidth=2)

            # Add scatter points
            plt.scatter(month_data["date"], month_data[f"{linkno}_gauge"], color='blue', edgecolor='black', s=50,
                        label='_nolegend_')
            plt.scatter(month_data["date"], month_data[f"{linkno}_geoglows"], color='orange', edgecolor='black', s=50,
                        label='_nolegend_')

            # Add titles, labels, and legend
            month_name = month_data["date"].dt.strftime('%B').iloc[0]  # Get month name
            plt.title(f'Comparison of Gauge and GHM HydroSOS Categories of {linkno} for {month_name}', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Values', fontsize=12)
            # Update y-axis labels
            y_labels = {1: "Low Flow", 2: "Below Normal", 3: "Normal", 4: "Above Normal", 5: "High Flow"}
            plt.yticks(ticks=list(y_labels.keys()), labels=list(y_labels.values()))
            plt.legend()
            plt.grid(True)
            plt.savefig(f'/Users/rachel1/Downloads/rachel nile/heatmaps/comparison_{linkno}_{month_name}.png', dpi=500, bbox_inches='tight')

            # Show the plot
            plt.show()

        monthly_abs_diff_df = pd.DataFrame(monthly_abs_diff)
        #sns.barplot(ax=axes[i], data=monthly_abs_diff_df, x="Month", y="Total_Abs_Diff")
        #axes[i].set_title(f"{linkno}", fontsize=14)
        #axes[i].set_xlabel("Month", fontsize=12)
        #axes[i].set_xticks(range(12))
        #axes[i].set_xticklabels(
         #   ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          #   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
           # rotation=45,
           # fontsize=10
        #)
        #if i == 0:
         #   axes[i].set_ylabel("Absolute Difference", fontsize=12)

        filtered_df = results.dropna(subset=[f"{linkno}_gauge", f"{linkno}_geoglows"])
        # Create the contingency table
        # Create the contingency table
        contingency_table = pd.crosstab(
            filtered_df[f"{linkno}_gauge"],
            filtered_df[f"{linkno}_geoglows"]
        )

        # Normalize to percentages (row-wise)
        contingency_table = contingency_table / contingency_table.values.sum()

        # Map categories to descriptive labels
        category_mapping = {
            1: "Low Flow",
            2: "Below Normal",
            3: "Normal",
            4: "Above Normal",
            5: "High Flow"
        }

        # Rename the index (gauge categories) and columns (modeled categories)
        #contingency_table.index = contingency_table.index.map(category_mapping)
        #contingency_table.columns = contingency_table.columns.map(category_mapping)
        #print(contingency_table)
        #plt.figure(figsize=(8, 6))
        #sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", fmt=".1f", cbar=True)
        #plt.title(f"Comparison of Observed and GHM Data for {linkno}")
        #plt.gca().xaxis.tick_top()
        #plt.xlabel("GHM Data")
        #plt.ylabel("Observed Data")
        # filename = f"/Users/rachel1/Downloads/rachel nile/heatmaps/heatmap_comparison_link_{linkno}.png"
        # plt.savefig(filename, dpi=500, bbox_inches="tight")
        #plt.show()
        # chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Results
        # print(f"Chi-Square Statistic: {chi2}")
        # print(f"p-value: {p}")
        # print(f"Degrees of Freedom: {dof}")
        # print("Expected Frequencies:")
        # print(expected)
        # Compute Cohen's Kappa
        # kappa = cohen_kappa_score(filtered_df[f"{linkno}_gauge"], filtered_df[f"{linkno}_geoglows"])
        # print(f"Cohen's Kappa: {kappa:.2f}")
        #heatmaps.append(contingency_table)
        #titles.append(f"GHM ID {linkno}")
#plt.tight_layout()
#plt.suptitle("Average Absolute Difference by Month for Each Gauge Station", fontsize=16, y=1.05)
#combined_filename = "/Users/rachel1/Downloads/rachel nile/heatmaps/abs_diff.png"
#plt.savefig(combined_filename, dpi=500, bbox_inches="tight")
#plt.show()

"""
n = len(heatmaps)
cols =3  # Number of columns in the grid
rows = (n + cols - 1) // cols  # Calculate number of rows needed

fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 6))
axes = axes.flatten()

# Compute global vmin and vmax
all_values = np.concatenate([heatmap.values.flatten() for heatmap in heatmaps])
vmin, vmax = np.min(all_values), np.max(all_values)

for i, (heatmap, title) in enumerate(zip(heatmaps, titles)):
    sns.heatmap(
        heatmap,
        annot=True,
        cmap="YlGnBu",
        fmt="0.2f",
        cbar=True,
        ax=axes[i],
        vmin=vmin,
        vmax=vmax
    )
    axes[i].set_title(title)
    axes[i].xaxis.tick_top()
    axes[i].set_ylabel("Observed Data")
    axes[i].set_xlabel("GHM Data")

# Remove unused axes if any
for j in range(i + 1, len(axes)):
   fig.delaxes(axes[j])

fig.suptitle("Heatmap for Comparison between GHM and Observed Data HydroSOS Categories", fontsize=16, y=0.95)

# Adjust layout with more vertical spacing
plt.tight_layout(h_pad=3.0)  # Increase vertical spacing between plots
combined_filename = "/Users/rachel1/Downloads/rachel nile/heatmaps/all_heatmaps.png"
plt.savefig(combined_filename, dpi=500, bbox_inches="tight")
#plt.show()
"""
