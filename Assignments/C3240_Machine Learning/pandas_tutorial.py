import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Processing Data with Pandas

# read in data from the file "air_temp.csv" and store it
# in the DataFrame "df"

df = pd.read_csv('air_temp.csv')

# print the first 5 weather recordings in the DataFrame `df`

df.head(5)

# print a concise summary of a DataFrame including the index dtype and columns, non-null values and memory usage

df.info()

df = df.dropna(axis=0)
data = df.assign(date=df["Year"].astype(str) + '-' + df["m"].astype(str) + '-' + df["d"].astype(str))
data = data.drop(['Year', 'm', 'd', 'Time zone'], axis=1)
data.head(10)
# data = data[['date','Time','Air temperature (degC)']]
# data.head(10)

# change column names

df.columns = ['year', 'month', 'day', 'time', 'time_zone', 'temperature']

# remove rows from dataframe "df" which contain missing values

df = df.dropna(axis=0)  # rows are considered as axis 0

# concatenate the 3 columns "year", "month", "day" into a new column "date" in format e.g. 2022-01-26

data = df.assign(date=df["year"].astype(str) + '-' + df["month"].astype(str) + '-' + df["day"].astype(str))

# remove columns "year", "month", "day", "time_zone" that are not used

data = data.drop(['year', 'month', 'day', 'time_zone'], axis=1)  # columns are axis 1

# switch column order

data = data[['date', 'time', 'temperature']]

# print the last 5 weather recordings of the new dataframe

data.tail(5)

# print some summary statistics of the rows in "data", such as mean, std, etc.

df.describe()

# Let us select only the column "temperature" of the DataFrame "data"
tmp = data['temperature']

print(type(tmp), '\n')  # check the type of this object
print(tmp)

data['temperature'].to_numpy()  # extract the values stored in a specific column into a ndarray

# select the first weather recording (row) stored in the DataFrame "data"
firstrow = data.iloc[0]  # `0` is the index of the first row

print("The first row: \n",firstrow)

# select the row with row label name `3` by using data.loc[ ]
# NOTE `3` is interpreted as a row label name , not an integer position along the index
# the row label name could be string or other data type, not only int
rowName3 = data.loc[3]
print("\n The row with row label name '3': \n",rowName3)


# we can select a subset of a DataFrame on some condition and create a new DataFrame

# create a "newdataset" which consists only of weather recordings in "data" at "time" `03:00`
newdataset= data[data['time'] == '03:00'] ;

# print randomly selected five weather recordings (rows) of "newdataset"
newdataset.sample(5)

# Demo Consider the weather observations recorded in air_temp.csv and loaded into the dataframe data. Let us now demonstrate how to define datapoints,
# their features and labels based on these weather observations. It is important to note that the choice (definition) of datapoints,
# their features and labels are design choices.
#
# We like to define a datapoint to represent an entire day, e.g.,
#
#     first datapoint represents the day 2021-06-01,
#     second datapoint represents the day 2021-06-02,
#     third datapoint represents the day 2021-06-03,
#     ...
#
# The total number 𝑚
#
# of datapoints is the number of days for which data contains weather recordings for daytime 01:00 and 12:00.
#
# We characterize the 𝑖
#
# -th datapoint (day) using
#
#     the temperature recorded at 01:00 during the 𝑖
#
# th day as its feature 𝑥(𝑖)
# the temperature recorded at 12:00 during the 𝑖
# th day as its label 𝑦(𝑖)
#
# We store the feature values 𝑥(𝑖),𝑖=1,…,𝑚
# in a (two-dimensional) numpy array X_demo with shape (m,1). The feature value 𝑥(𝑖) is stored in the entry X_demo[i-1,0] (note that indexing of
# numpy arrays starts with 0!). The label values 𝑦(𝑖),𝑖=1,…,𝑚 are stored in a (one-dimensional) numpy array y_demo with shape (m,). Finally,
# we generate a scatterplot where the 𝑖th datapoint is depicted by a dot located at coordinates (𝑥(𝑖),𝑦(𝑖))
#
# .
#
# HINT: Reshape X_demo into a 2-D array by using array.reshape(-1, 1). This asks numpy to make the second dimension
# length one and automatically calculate the needed length of the first dimension so that the feature fits in the container which expects
# a 2-D array. (e.g.,the .fit() method of LinearRegression)

# create a list containing the dates for which at least one recording is contained in "data"
dates = data['date'].unique()

features = []  # list used for storing features of datapoints
labels = []  # list used for storing labels of datapoints

m = 0  # number of datapoints created so far

# iterate through the list of dates for which we have weather recordings
for date in dates:
    datapoint = data[(data['date'] == date)]  # select weather recordinds corresponding at day "date"
    row_f = datapoint[(datapoint.time == '01:00')]  # select weather recording at time "01:00"
    row_l = datapoint[(datapoint.time == '12:00')]  # select weather recording at time "12:00"
    if len(row_f) == 1 and len(row_l) == 1:
        feature = row_f['temperature'].to_numpy()[0]  # extract the temperature recording at "01:00" as feature
        label = row_l['temperature'].to_numpy()[0]  # extract the temperature recording at "12:00" as label
        features.append(feature)  # add feature to list "features"
        labels.append(label)  # add label to list "labels"
        m = m + 1

X_demo = np.array(features).reshape(m, 1)  # convert a list of len=m to a ndarray and reshape it to (m,1)
y_demo = np.array(labels)  # convert a list of len=m to a ndarray

print("number of datapoints:", m)
print("the shape of the feature matrix is: ", X_demo.shape)
print('the shape of the label vector is: ', y_demo.shape)

# visualize the datapoints
fig = plt.figure()  # create a figure

ax = fig.add_subplot(1, 1, 1)  # add an axes object to the figure

ax.scatter(X_demo, y_demo)  # plot a scatterplot in the axes to visualize the datapoints
ax.set_xlabel('Temperature at 01:00')  # set the label of x-axis
ax.set_ylabel('Temperature at 12:00')  #
ax.set_title('Tmp_01:00 vs Tmp_12:00')

plt.show()

# one line of code `plt.scatter(X_demo,y_demo)` without creating figure and axes objects
# can also realize a scatter plot, but it's worth getting yourself familiar with the relation among figure, axes and plot

