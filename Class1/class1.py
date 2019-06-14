import pandas as pd
import numpy as np
step_data = [3620,7891,9761,3907,4338,5373]

step_counts = pd.Series(step_data,name = 'steps')

# to point a date for each data which we have start from 2015-03-29
step_counts.index = pd.date_range('20150329' , periods = 6)

print(step_counts)
# like a dictionary you can call it eather by date, or by index number of column
print(step_counts['2015-04-02'])
print(step_counts[4])
# select all of April
print(step_counts['2015-04'])
# view the data type
print(step_counts.dtypes)
# convert to a float
step_counts = step_counts.astype(np.float)
# view the type
print(step_counts.dtypes)

#create invalid date
step_counts[1:4] = np.NaN
print(step_counts)
# now fill it with zeros
step_counts = step_counts.fillna(0.)
# or equivalaently ::
#step_counts.fillna(0.,inplace = True)
print(step_counts)


### Data Frame creation with pandas

# Cycling distance
cycling_data = [10.7 , 0 , None, 2.4 , 15.3, 10.9 , 0 , None]

# Create a tuple of data
joined_data = list(zip(step_data,cycling_data))
#print("joined_data is : ",joined_data)

# creating the dataframe from joined data
activity_df = pd.DataFrame(joined_data)
print(activity_df)

# Add Column names to dataframe
activity_df = pd.DataFrame(joined_data, index = pd.date_range('20150329',periods = 6),columns = ['Walking','Cycling'])
print(activity_df)

# select row of data by index name
print(activity_df.loc['2015-04-01'])

# select row of data by integer position
print(activity_df.iloc[-3]) # it start from "0" but if we like to count from the end it will show us the 3rd column of last columns

# Name of column
print(activity_df['Walking'])

# object oriented approch
print(activity_df.Walking)

# first column
print("first column is:",activity_df.iloc[:,0])

###
# location of data
filepath = '/Users/MAHDI/Desktop/MACHINE_LEARNING_INTEL/Intel-ML101_Class1/data/Iris_Data.csv'
# import this data
data = pd.read_csv(filepath)

#print a few rows
print(data.iloc[:5])

# Create a new column that is a product of both measurements
data['sepal_area'] = data.sepal_length * data.sepal_width

# Print a few rows and columns  (first 5 rows of last 3 columns)
print(data.iloc[:5, -3:])

# The lambda function applies what follows it to each row of data
# species is one of the columns of our df
data['abbrev'] = (data.species.apply(lambda x: x.replace('Iris-','')))

print(data.iloc[:5, -3:])

# concatenate the first two and last two rows
small_data = pd.concat([data.iloc[:2], data.iloc[-2:]])

print(small_data.iloc[:,-3:])

# See the "join" method for SQL style joining of dataframes

## use the size method with a DataFrame to get count for a series, use the .value_counts method
group_sizes = (data.groupby('species').size())
print(group_sizes)


### Mean calculated on a DataFrame
print(data.mean())

# Median Calculation on a Series
print(data.petal_length.median())

# mode calculation on a series
print(data.petal_length.mode())

# Standard dev, variance, and SEM
print(data.petal_length.std(),
      data.petal_length.var(),
      data.petal_length.sem())

# As well as quantiles
print(data.quantile(0))

print(data.describe())

# Sample 5 rows without replacement
# Dataframe can be randomly sampled
sample = (data.sample(n = 5,replace = False, random_state = 42))

print(sample.iloc[:,-3:])


##### VISUALISATION ##### with matplotlib, Pandas, Seaborn

import matplotlib.pyplot as plt

# select the data which we like to show, tyoe of market and lable
plt.plot(data.sepal_length,data.sepal_width,ls='', marker='o', label='sepal')
# we always should write plt.show to show the results
plt.show()
# if we dont  write it imeiately we will have the superposition of data in one figure

plt.plot(data.petal_length,data.petal_width,ls='', marker='o', label='petal')

plt.show()

plt.hist(data.sepal_length , bins = 25)
plt.show()

# custemize the feature of Matplotlib plots
fig , ax =plt.subplots()


# arange a barh with 10 cathegorize and add the value of sepal_width to it(first 10 raw)
ax.barh(np.arange(10), data.sepal_width.iloc[:10])


# set position of tickes and tick labels


# here we want the graph to show the range of change in y in a position 0f "0.4" to "10.4" one by one
ax.set_yticks(np.arange(0.0, 10.0, 1.0))
# here we asak the graph to consider number from 1 to 10 for y variable
ax.set_yticklabels(np.arange(1,11))
ax.set(xlabel = 'xlabel', ylabel = 'ylabel', title = 'title')
plt.show()

# Apply Statistical Calculation
(data.groupby('species')
 .mean()
 .plot(color = ['red','blue','black','green'],
       fontsize = 10.0 , figsize = (4,4)))
plt.show()

# use seaborn to make join distribution and scotter plot
import seaborn as sns

# Draw a plot of two variables with bivariate and univariate graphs
# kind : will add the type fo feature we like to have in our plot such as "reg, scatter, resid, kde,hex"
sns.jointplot(x='sepal_length',y='sepal_width',data = data, size = 4, kind = "reg")
plt.show()

# Draw a scatter plot , then add a joint density

g = (sns.jointplot('sepal_length','sepal_width', data = data, color = 'k').plot_joint(sns.kdeplot,zorder = 0, n_level= 6))
plt.show()


# Correlation plot of all variables pairs can also be done with Seaborn
sns.pairplot(data, hue = 'species' , size = 3)
plt.show()
