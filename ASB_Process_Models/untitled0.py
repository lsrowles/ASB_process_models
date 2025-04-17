# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:33:21 2024

@author: waqar
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# Load the Excel file
file_path = "C:\Important\Master file GW DatA.xlsx"

# Load the 'NOx' sheet into a DataFrame
df_nox = pd.read_excel(file_path, sheet_name='NOx')

# Display basic information and check for missing values
print(df_nox.info())
print(df_nox.isnull().sum())

# Visualize the distribution of the 'Results' variable
plt.figure(figsize=(8, 6))
sns.histplot(df_nox['Results'], bins=10, kde=True)
plt.title('Distribution of NOx Results')
plt.xlabel('Results (mg/L)')
plt.ylabel('Frequency')
plt.show()

# Visualize the relationship between 'Sampling point' and 'Results'
plt.figure(figsize=(8, 6))
sns.boxplot(x='Sampling point', y='Results', data=df_nox)
plt.title('NOx Results by Sampling Point')
plt.xlabel('Sampling Point')
plt.ylabel('Results (mg/L)')
plt.xticks(rotation=45)
plt.show()

# Prepare the data for ANOVA by binning 'Sampling point'
df_nox['Sampling_point_bins'] = pd.qcut(df_nox['Sampling point'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

# Perform the ANOVA
anova_model = smf.ols(formula='Results ~ C(Sampling_point_bins)', data=df_nox).fit()
print(anova_model.summary())

# Calculate the mean and standard deviation for the groups
mean_sd_df = df_nox.groupby('Sampling_point_bins')['Results'].agg(['mean', 'std'])
print(mean_sd_df)


import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = "C:\Important\Master file GW DatA.xlsx"
excel_data = pd.ExcelFile(file_path)

# Load the data from the 'NOx' sheet
nox_data = pd.read_excel(excel_data, sheet_name='NOx')

# Basic descriptive statistics
descriptive_stats = nox_data['Results'].describe()
print(descriptive_stats)

# Create a figure for the distribution of NOx levels
plt.figure(figsize=(10, 6))

# Histogram of NOx concentration levels
plt.hist(nox_data['Results'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of NOx Concentration Levels')
plt.xlabel('NOx Concentration (mg/L)')
plt.ylabel('Frequency')

# Show the plot
plt.grid(True)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Corrected Sample Data
data = {
    "Transect": ["ISC_GW_1"] * 23,
    "Sampling": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
    "Sample Date": ["202011", "202105", "202107", "20210306", "20211108", "202110", 
                    "202011", "202105", "202107", "20210306", "20211108", "202110", 
                    "202105", "202107", "20210306", "20211108", "202110", "202011", 
                    "202105", "202107", "20210306", "20211108", "202110"],
    "Test": ["NOx_H2O_5"] * 23,
    "Results": [0.076, 0.008, 0.016, 0.09, 0.034, 0.724, 
                0.29, 0.055, 0.01, 0.002, 0.055, 0.096, 
                0.075, 0, 0.016, 0.023, 0.025, 0.229, 
                0.031, 0.024, -0.004, 0.009, 0.006],
    "Units": ["mg/L"] * 23
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert the 'Sample Date' to datetime format for better plotting
df['Sample Date'] = pd.to_datetime(df['Sample Date'], format='%Y%m%d', errors='coerce')

# Plotting
plt.figure(figsize=(10, 6))

# Plot each sampling point
for sampling_point in df['Sampling'].unique():
    subset = df[df['Sampling'] == sampling_point]
    plt.plot(subset['Sample Date'], subset['Results'], marker='o', label=f'Sampling Point {sampling_point}')

# Add labels and title
plt.xlabel('Sample Date')
plt.ylabel('NOx Concentration (mg/L)')
plt.title('NOx Concentration Over Time by Sampling Point')
plt.legend(title='Sampling Points')

# Show the plot
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
data = {
    'Transect': ['ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1',
                 'ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1',
                 'ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1',
                 'ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1', 'ISC_GW_1'],
    'Sampling': [1, 1, 1, 1, 1, 1, 
                 2, 2, 2, 2, 2, 2,
                 3, 3, 3, 3, 3,
                 4, 4, 4, 4, 4],
    'Sample Details': ['202011', '202105', '202107', '20210306', '20211108', '202110',
                       '202011', '202105', '202107', '20210306', '20211108', '202110',
                       '202105', '202107', '20210306', '20211108', '202110',
                       '202011', '202105', '202107', '20210306', '20211108', '202110'],
    'Results': [0.076, 0.008, 0.016, 0.09, 0.034, 0.724,
                0.29, 0.055, 0.01, 0.002, 0.055, 0.096,
                0.075, 0, 0.016, 0.023, 0.025,
                0.229, 0.031, 0.024, -0.004, 0.009, 0.006],
    'Units': ['mg/L'] * 22
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert the Sample Details to datetime for plotting
df['Sample Details'] = pd.to_datetime(df['Sample Details'], format='%Y%m%d', errors='coerce').fillna(pd.to_datetime(df['Sample Details'], format='%Y%m'))

# Plotting
plt.figure(figsize=(12, 8))
sns.boxplot(x='Sample Details', y='Results', hue='Sampling', data=df)
plt.title('NOx Values at Different Sampling Points Over Time')
plt.xlabel('Sample Dates')
plt.ylabel('NOx (mg/L)')
plt.legend(title='Sampling Points')
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Example data creation
data = {
    'Transect': ['L1'] * 24 + ['L2'] * 24 + ['L3'] * 24 + ['L4'] * 24,
    'Sampling_Point': (['Point1', 'Point2', 'Point3', 'Point4'] * 6) * 4,
    'Date': ['202011', '202105', '202107', '20210306', '20211108', '202110'] * 16,
    'NOx': [0.076, 0.29, 0.229, 0.008, 0.055, 0.075, 0.031, 0.016, 0.01, 0, 0.024, 0.09,
            0.039, 0.054, 0.057, 0.006, 0.006, 0.033, 0.051, 0.03, 0.006, -0.004, 0.113, 0.023,
            0.029, 0.325, 1.272, 0.137, 0.003, 0.684, 0.008, 0.032, 0.039, 0.067, 0.003, 0.061,
            0.315, 0.282, 0.177, 0.077, 0.016, 0.018, 0.053, 0.088, 0.017, 0.022, 0.012, 0.013,
            0.029, 0.325, 1.272, 0.137, 0.003, 0.684, 0.008, 0.032, 0.039, 0.067, 0.003, 0.061,
            0.315, 0.282, 0.177, 0.077, 0.016, 0.018, 0.053, 0.088, 0.017, 0.022, 0.012, 0.013]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Convert 'Transect', 'Sampling_Point', and 'Date' to categorical variables
df['Transect'] = pd.Categorical(df['Transect'])
df['Sampling_Point'] = pd.Categorical(df['Sampling_Point'])
df['Date'] = pd.Categorical(df['Date'])

# Perform ANOVA
model = ols('NOx ~ C(Transect) * C(Date)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Tukey HSD test if needed
mc = sm.stats.multicomp.MultiComparison(df['NOx'], df['Transect'])
tukey_result = mc.tukeyhsd()
print(tukey_result)

# Plotting
plt.figure(figsize=(14, 8))

# Create a boxplot for NOx concentrations by Transect and Date
sns.boxplot(x='Date', y='NOx', hue='Sampling_Point', data=df)
plt.title('NOx Concentration by Date and Sampling Point')
plt.ylabel('NOx Concentration (mg/L)')
plt.xlabel('Date')
plt.legend(title='Sampling Point', loc='upper right')
plt.show()

# Create scatter plots with different colors for sampling points
g = sns.FacetGrid(df, col="Transect", hue="Sampling_Point", height=5, aspect=1)
g.map(sns.scatterplot, "Date", "NOx")
g.add_legend()
plt.subplots_adjust(top=0.85)
g.fig.suptitle('NOx Concentration by Transect over Time')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data organized in a dictionary for easy DataFrame creation
data = {
    'Transect': ['L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1',
                 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2', 'L2',
                 'L3', 'L3', 'L3', 'L3', 'L3', 'L3', 'L3', 'L3', 'L3', 'L3', 'L3', 'L3',
                 'L4', 'L4', 'L4', 'L4', 'L4', 'L4', 'L4', 'L4', 'L4', 'L4', 'L4', 'L4'],
    'Sampling Point': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
    'Date': ['202011', '202011', '202011', '202011', '202105', '202105', '202105', '202105',
             '202107', '202107', '202107', '202107',
             '202011', '202011', '202011', '202011', '202105', '202105', '202105', '202105',
             '202107', '202107', '202107', '202107',
             '202011', '202011', '202011', '202011', '202105', '202105', '202105', '202105',
             '202107', '202107', '202107', '202107',
             '202105', '202105', '202105', '202105', '202107', '202107', '202107', '202107',
             '20210306', '20210306', '20210306', '20210306'],
    'NOx': [0.076, 0.29, 0.229, 0.008, 0.055, 0.075, 0.031, 0.016, 0.01, 0, 0.024, 0.09,
            0.039, 0.054, 0.057, 0.006, 0.006, 0.033, 0.051, 0.03, 0.006, -0.004, 0.113, 0.023,
            0.029, 0.325, 1.272, 0.137, 0.003, 0.684, 0.008, 0.032, 0.039, 0.067, 0.003, 0.061,
            0.008, 0.097, 0.167, -0.001, 0.055, 0.019, 0.03, 0.024, 0.306, 0.789, 0.009, 0.027]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Set up the plot with 4 subplots, one for each location
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
fig.suptitle('NOx Concentration Over Time for Different Sampling Points', fontsize=16)

# Plotting for each location
locations = df['Transect'].unique()
colors = sns.color_palette("husl", 4)

for i, location in enumerate(locations):
    ax = axs[i // 2, i % 2]  # Determine subplot position
    location_data = df[df['Transect'] == location]
    
    for j, point in enumerate(location_data['Sampling Point'].unique()):
        point_data = location_data[location_data['Sampling Point'] == point]
        ax.scatter(point_data['Date'], point_data['NOx'], label=f'Point {point}', color=colors[j], s=100)
    
    ax.set_title(f'Location {location}')
    ax.set_xlabel('Date')
    ax.set_ylabel('NOx Concentration (mg/L)')
    ax.legend()

# Adjust the layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Display the plot
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example data generation
np.random.seed(42)

# Define the structure of the data
locations = ['Location_1', 'Location_2', 'Location_3', 'Location_4']
sampling_points = ['Point_1', 'Point_2', 'Point_3', 'Point_4']
dates = pd.date_range(start='2023-01-01', periods=6, freq='M')

# Generate random NOx data
data = []
for loc in locations:
    for point in sampling_points:
        nox_values = np.random.rand(len(dates)) * 100  # Random NOx values between 0 and 100
        for date, nox in zip(dates, nox_values):
            data.append([loc, point, date, nox])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Location', 'Sampling_Point', 'Date', 'NOx'])

# Statistical analysis (basic descriptive stats)
stats = df.groupby(['Location', 'Sampling_Point'])['NOx'].describe()
print(stats)

# Plotting
plt.figure(figsize=(14, 10))
sns.set(style="whitegrid")

# Create subplots for each location
for i, loc in enumerate(locations):
    plt.subplot(2, 2, i+1)
    subset = df[df['Location'] == loc]
    sns.scatterplot(data=subset, x='Date', y='NOx', hue='Sampling_Point', palette='deep')
    plt.title(loc)
    plt.xlabel('Date')
    plt.ylabel('NOx Concentration')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)  # Assuming NOx values are between 0 and 100

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Example DataFrame
data = {
    'Location': np.repeat(['L1', 'L2', 'L3', 'L4'], 16),
    'Sampling Point': np.tile(np.repeat(['P1', 'P2', 'P3', 'P4'], 4), 4),
    'Date': np.tile(['2024-01-01', '2024-01-15', '2024-02-01', '2024-02-15'], 16),
    'Measurement': np.random.rand(64) * 100
}

df = pd.DataFrame(data)

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Add a time variable for regression analysis
df['Time'] = (df['Date'] - df['Date'].min()).dt.days

# Perform a simple linear regression for each Location and Sampling Point
regression_results = df.groupby(['Location', 'Sampling Point']).apply(
    lambda x: sm.OLS(x['Measurement'], sm.add_constant(x['Time'])).fit()
)

# Plotting with Seaborn
g = sns.FacetGrid(df, row='Location', col='Sampling Point', margin_titles=True)
g.map(sns.scatterplot, 'Date', 'Measurement')

# Add regression lines and annotations
for ax, (location, sampling_point), result in zip(g.axes.flatten(), regression_results.index, regression_results):
    # Plot regression line
    x_vals = np.array(ax.get_xlim())
    y_vals = result.params[0] + result.params[1] * (x_vals - df['Date'].min().timestamp()/86400)
    ax.plot(x_vals, y_vals, '--', color='red')

    # Annotate with regression equation and R-squared
    eqn = f"$y = {result.params[1]:.2f}x + {result.params[0]:.2f}$"
    r_squared = f"$R^2 = {result.rsquared:.2f}$"
    ax.text(0.05, 0.9, f"{eqn}\n{r_squared}", transform=ax.transAxes, color='red')

# Adjusting the plot
g.set_axis_labels('Date', 'Measurement Value')
g.set_titles(col_template="{col_name}", row_template="{row_name}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Example DataFrame
data = {
    'Location': np.repeat(['L1', 'L2', 'L3', 'L4'], 16),
    'Sampling Point': np.tile(np.repeat(['P1', 'P2', 'P3', 'P4'], 4), 4),
    'Date': np.tile(['2024-01-01', '2024-01-15', '2024-02-01', '2024-02-15'], 16),
    'Measurement': np.random.rand(64) * 100
}

df = pd.DataFrame(data)

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Plotting with Seaborn
g = sns.FacetGrid(df, row='Location', col='Sampling Point', margin_titles=True)
g.map(sns.scatterplot, 'Date', 'Measurement')

# Adjusting the plot
g.set_axis_labels('Date', 'Measurement Value')
g.set_titles(col_template="{col_name}", row_template="{row_name}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load the data from Excel
file_path = "C:\Important\Master file GW DatA.xlsx"
nox_data = pd.read_excel(file_path, sheet_name='NOx')

# Plotting NOx levels across different location patches and sampling points over time
plt.figure(figsize=(12, 6))

# Create a boxplot
sns.boxplot(x='Transect', y='Results', hue='Sample Details', data=nox_data)

# Customize the plot
plt.title('NOx Levels by Location Patch and Sampling Date')
plt.xlabel('Location Patch')
plt.ylabel('NOx Levels (mg/L)')
plt.legend(title='Sample Date', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

# Perform ANOVA
anova_results = stats.f_oneway(*[group['Results'].values for name, group in nox_data.groupby(['Transect', 'Sample Details'])])

print('ANOVA results:', anova_results)


import pandas as pd

# Load the Excel file
file_path = "C:\Important\Master file GW DatA.xlsx"
nox_data = pd.read_excel(file_path, sheet_name='NOx')

# Add a new column to identify land patches
nox_data['Land_Patch'] = nox_data['Transect'].str.extract(r'^(ISC_GW_\d)_\d')

# Group by Land_Patch and Sample Details (Date)
nox_grouped = nox_data.groupby(['Land_Patch', 'Sample Details'])['Results'].mean().reset_index()

# Display the organized data
print(nox_grouped.head())

from scipy.stats import f_oneway

# Perform ANOVA to compare NOx levels between different land patches
land_patches = nox_grouped['Land_Patch'].unique()
anova_data = [nox_grouped[nox_grouped['Land_Patch'] == lp]['Results'] for lp in land_patches]
f_stat, p_value = f_oneway(*anova_data)

print(f"ANOVA F-statistic: {f_stat}, P-value: {p_value}")

import matplotlib.pyplot as plt
import seaborn as sns

# Plot NOx concentrations over time for each land patch
plt.figure(figsize=(12, 8))
sns.lineplot(x='Sample Details', y='Results', hue='Land_Patch', data=nox_grouped, marker="o")
plt.title('NOx Concentrations Over Time by Land Patch')
plt.xlabel('Date')
plt.ylabel('NOx Concentration (mg/L)')
plt.legend(title='Land Patch')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


