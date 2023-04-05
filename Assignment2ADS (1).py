# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 18:15:49 2023

@author: Yeshu
"""

# Importing the required libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np


def world_data(filedata):
    """
    Reads a CSV file and returns two dataframes.

    Parameters:
    -----------
    dataset : str
        The path to the CSV file.

    Returns:
    --------
    dt_name : pandas.DataFrame
        A dataframe with the country names as index and the columns as years.
    dt_data_country : pandas.DataFrame
        A dataframe with the country codes as index and the columns as years.
    """
    # read the CSV file and skip the first 4 rows of metadata
    info = pd.read_csv(filedata, skiprows=4)

    # drop unnecessary columns from the dataframe
    countries = info.drop(
        columns=['Country Code', 'Indicator Code', 'Unnamed: 66'], inplace=True)

    # set the index of the dataframe to 'Country Name' and transpose the dataframe
    countries = info.set_index('Country Name').T

    # set the index of the dataframe to 'Country Name' and reset the index
    df_name = info.set_index('Country Name').reset_index()
    return df_name, countries


def attribute(indicators, info):
    """
    Returns a dataframe with the specified indicator.

    Parameters:
    -----------
    indicators : str
        The name of the indicator to select.
    dt : pandas.DataFrame
        The dataframe to select from.

    Returns:
    --------
    dt : pandas.DataFrame
        A dataframe with the specified indicator.
    """
    info = info[info['Indicator Name'].isin([indicators])]
    return info


def choose_country(countries, info):
    """
    Returns a dataframe with the specified country.

    Parameters:
    -----------
    country : str
        The name of the country to select.
    dt : pandas.DataFrame
        The dataframe to select from.

    Returns:
    --------
    dt : pandas.DataFrame
        A dataframe with the specified country.
    """
    # Selecting data for the specified countries
    info = info[info['Country Name'].isin([countries])]

    # Dropping unnecessary column and setting index to Indicator Name
    info = info.set_index("Indicator Name")
    info = info.drop("Country Name", axis=1)

    # Transposing the dataframe
    info = info.T
    return info


# Reading the CSV file
df_name, countries = world_data(r"C:\Users\yeshwanth\OneDrive\Desktop\ADS2\wbdata.csv")
print(df_name.describe())
print(countries.describe())


#bar plot
# Creating a list of countries
countries1 = ['Argentina', 'Australia', 'Brazil', 'Canada', 'China', 'India',  'Sri Lanka']

# Selecting data for these countries and years
pop = df_name.loc[(df_name['Country Name'].isin(countries1)) & (df_name['Indicator Name'] == 'Population growth (annual %)') & (
    df_name[['1970', '1980', '1990', '2000', '2010', '2020']].notnull().all(axis=1)), :]

# Grouping the data by country and year
pop_grouped = pop.groupby(
    'Country Name')[['1970', '1980', '1990', '2000', '2010', '2020']].agg(list)
print(pop_grouped)

# Creating a multiple bar plot
fig, ax = plt.subplots()
ind = np.arange(len(countries1))
width = 0.13

# Create a list of years to plot
years = ['1970', '1980', '1990', '2000', '2010', '2020']

# Calculate the x coordinates of the bars for each year
x_coords = np.arange(len(countries1))
x_offsets = np.arange(len(years)) - 3
x_coords = x_coords[:, None] + x_offsets[None, :] * width

# Extract the data for each year
data = [pop_grouped[year].apply(lambda x: x[0]) for year in years]

# Plot the bars for each year
for i, year in enumerate(years):
    ax.bar(x_coords[:, i], data[i], width, label=year,
            alpha=1, edgecolor='black')

# Set the axis labels and title
ax.set_xticks(x_coords.mean(axis=1))
ax.set_xticklabels(countries1, rotation=90)
ax.set_ylabel('% of population')
ax.set_title('Population growth (annual %)', fontsize="10")

# Add the legend
ax.legend(fontsize="7", loc ="upper right")

plt.show()

#another bar plot
# Creating a list of countries
countries1 = ['Australia', 'Canada', 'China', 'India']

# Selecting data for these countries and years, removing rows with missing values
pop = df_name.loc[(df_name['Country Name'].isin(countries1)) & (df_name['Indicator Name'] == 'Urban population (% of total population)') & (
    df_name[[ '1990', '2000', '2010', '2020']].notnull().all(axis=1)) & (
    df_name[['1990', '2000', '2010', '2020']].notnull().any(axis=1)), :].dropna()

# Grouping the data by country and year
pop_grouped = pop.groupby(
    'Country Name')[['1990', '2000', '2010', '2020']].agg(list)
print(pop_grouped)

# Creating a multiple bar plot
fig, ax = plt.subplots()
ind = np.arange(len(countries1))
width = 0.13

# Create a list of years to plot
years = [ '1990', '2000', '2010', '2020']

# Calculate the x coordinates of the bars for each year
x_coords = np.arange(len(countries1))
x_offsets = np.arange(len(years)) - 3
x_coords = x_coords[:, None] + x_offsets[None, :] * width

# Extract the data for each year
data = [pop_grouped[year].apply(lambda x: x[0]) for year in years]

# Plot the bars for each year
for i, year in enumerate(years):
    ax.bar(x_coords[:, i], data[i], width, label=year,
            alpha=1, edgecolor='black')

# Set the axis labels and title
ax.set_xticks(x_coords.mean(axis=1))
ax.set_xticklabels(countries1, rotation=90)
ax.set_ylabel('% of population')
ax.set_title('Urban population (% of total population)', fontsize="10")

# Add the legend
ax.legend(fontsize="7", loc ="upper right")

plt.show()


# Selecting data for a particular country using 'choose
elec_prod = choose_country('India', df_name)

# Selecting the relevant columns
percentage = elec_prod[['Electricity production from oil sources (% of total)',
                        'Electricity production from natural gas sources (% of total)',
                        'Electricity production from hydroelectric sources (% of total)',
                        'Electricity production from coal sources (% of total)']]

# Calculating the correlation matrix
correlation = percentage.corr()

# Creating a heatmap using seaborn library
sea = sb.heatmap(correlation, annot=True)

# Displaying plot
plt.show()

# Selecting data for a specific country
t_pop = choose_country('United Kingdom', df_name)

# Selecting columns of interest
pop_percentage = t_pop[['Urban population (% of total population)',
                        'Urban population', 'Urban population growth (annual %)',
                        'Population, total', 'Population growth (annual %)']]

# Calculating correlation between the selected columns
pop_correlation = pop_percentage.corr()

# Creating a heatmap to visualize the correlation matrix
sea = sb.heatmap(pop_correlation, annot=True)
plt.show()
