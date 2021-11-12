import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('train.csv')
del data['Cabin']
no_na_data = data.dropna(axis=0)    # Delete the rows containing missing data

# npData is defined here
npAges = np.array(no_na_data['Age'])
npFares = np.array(no_na_data['Fare'])


# ----------------------------------------------------------------------------------------------------------------------
# This section defines the function used in teh data set assessment. It is for modularity purposes.


# This function gets the total number of passengers in the titanic data set.
def get_total_num_passengers(dt):
    return len(dt)


# This function gets the number of survived passengers.
def get_num_survived(dt):
    survived = 0
    for survivor in dt['Survived']:
        if survivor == 0:
            survived += 1
    return survived


# This function gets the number of passengers that died.
def get_num_died():
    total_died = get_total_num_passengers(no_na_data) - get_num_survived(no_na_data)
    return total_died


# This function finds the missing values in a dataset.
def find_missing(dt):
    missing = dt.isnull().sum()
    return missing


# This function creates a dictionary with statistics of a data set: min, max, mean, median, & standard deviation.
def get_stats(dt):
    stats = {
        'min': np.amin(dt),
        'max': np.amax(dt),
        'mean': np.mean(dt),
        'median': np.median(dt),
        'std': np.std(dt)
    }
    return stats


# This function plots the distribution of values of a column in the data set.
# It can only be used for individual columns of the data set and not for the data set as a whole.
def plot_distribution(dt, column, title, color='b'):
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 6)
    sns.distplot(dt[column], color=color)
    pl.title(title, fontsize=16)
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# The bllow section is a print script to analyse different assumptions of the data set.


# Total num of passengers and features of the data set.
print('Number of passengers is {}.'.format((get_total_num_passengers(no_na_data))))
print('Number of features for each passenger is {}.'.format(len(data.columns)))
print(data.head(), '\n')


# Checking the number of missing values:
print('Number of missing values in each column is: \n', find_missing(no_na_data), '\n')


# Assessing the survived column
print('Number of passengers that survived was {}.'.format(get_num_survived(no_na_data)))
print('Number of passengers that died {}.'.format(get_num_died()))
print('% of passengers that survived is {}.\n'.format(100*(get_num_survived(no_na_data)/len(data))))


# Statistics on Age column
age_stats = get_stats(npAges)
print('Youngest passenger is {} years.'.format(age_stats['min']))
print('Oldest passenger is {} years.'.format(age_stats['max']))
print('Mean age is {} years.'.format(age_stats['mean']))
print('Median age is {} years.'.format(age_stats['median']))
print('Standard deviation is {} years.\n'.format(age_stats['std']))

# Youngest and oldest passengers
print(data[data['Age'] == 80])
print(data[data['Age'] == 0.42], '\n')


# Statistics on Fares column
fares_stats = get_stats(npFares)
fares_stats_filtered = get_stats(npFares[npFares != 0])
print('Cheapest ticket fare is £{}.'.format(round(fares_stats_filtered['min'], 2)))
print(('Most expensive ticket is £{}.'.format(round(fares_stats['max'], 2))))
print('Mean fare is £{}.'.format(round(fares_stats_filtered['mean'], 2)))
print('Median fare is £{}.'.format(round(fares_stats_filtered['median'], 2)))
print('SD of fare is £{}.\n'.format(round(fares_stats_filtered['std'], 2)))


# Plot distributions

# Distribution of ages:
plot_distribution(dt=no_na_data, column='Age', title='Distribution of Ages')

# Distribution of fares:
plot_distribution(dt=no_na_data, column='Fare', title='Distribution of Fares', color='r')

# Distribution of
plot_distribution(dt=no_na_data, column='Pclass', title='Distribution of Classes')


# Plot bar chart of classes:
sns.countplot(data['Pclass'], hue=data['Survived'])
pl.title('Number of survivors and non-survivors ', fontsize=16)
plt.show()
