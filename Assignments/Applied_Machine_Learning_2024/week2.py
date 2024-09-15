import pandas as pd
import matplotlib.pyplot as plt

# 2. Utilizing positioning data

# The accompanying dataset contains sensors data from a positioning system,
# attached to a Leverkusen junior soccer player. The purpose is to collect information about each player and the
# team for player and team development purposes. The data contains sensor readings
#
#    Lateral accelarations from Inertial Measuremnt Unit (IMU)
#    Angular accelerations from IMU
#    Latitude and longitude from Global Navigation Satellite System (GNSS)
#    And some other sensor and calculated data

# Some of the last columns do not contain any values. And some time in the beginning the GNSS positioning is not
# working, and therefore the latitude and longitude is available only after some time.
#
# Each row contains a sensor reading during a specific time. The time interval between the samples (=rows) is 1/100 s.
#
# It is easiest to read complex CSV-files with a sophisticated pandas read_csv() -function.

# 2.1. Read the data

# Examine the data in the editor, to see what it's structure is
# (CSV:s are sometimes rather badly formed, in this case it is quite Ok though)
# Read the data from the file 101666_1678179706_4556.csv to pandas dataframe called as D_all.
# Notice that there are a few first lines which are not in CSV form. They must be skipped.
# Check the first five rows of the data to see that you have correctly read it.
# Assign the number of rows of the whole dataset to variable n and number of columns to variable p
# pd.set_option('display.max_columns', None)

file = pd.read_csv("101666_1678179706_4556.csv", sep=",", header=3)
D_all = pd.DataFrame(file)
# [0,:] = first row & all columns,
# [:, 0] = all rows, first column,
# [0:3, 0] = first three,
# [2:, 0] = all rows starting from 2
# print(A[:5], "\n", A.iloc[0:5, :])

n = len(D_all)
p = len(D_all.columns)

# print(n, p)

# 2.2. Remove rows with missing coordinate data
# Study the data with the describe() -function.
# How many values there are in each column.
# Assign in variable nL, how many rows have data in Longitude -column.
# Drop each row, where the Longitude-value is missing. and assign the result to new data-frame D

# describe = D_all.describe()
nL = D_all["Longitude"].count()
D = D_all.dropna(subset=['Longitude'])

print(nL, D[['Acceleration.forward', 'Acceleration.side', 'Acceleration.up']])

# 2.2.0.1. Plot the data

# 1. Study the column names, by printing the D.columns variable
# 2. Plot the three accelerations: Acceleration.forward, Acceleration.side,
#    and Acceleration.up between rows (=samples) between 0-500 in the same figure. Use the D.plot() and assign
#    the resulting figure to the variable called fig, as follows: fig = D.plot(...
# 3. Define the label for y-axis. You can use LaTeX strings as follows 'Acceleration $\cdot 10 m/s^2$' .
#    The part between dollar signs is interpreted as mathematical expression.
# 4. Define also the label for x-axis. It is the time in centiseconds or 10 ms.

# TIP: It is easiest to plot Pandas dataframes by just using the build in plot() function of the dataframe object.
# Study its properties by writing D.plot and then hitting the Shift-TAB button combination. You should see the
# documentation of print

columns = D.columns
print(columns)
# fig = D[['Acceleration.forward', 'Acceleration.side', 'Acceleration.up']].iloc[0:500].plot()
# plt.ylabel('Acceleration $\\cdot 10 m/s^2$')
# plt.xlabel('Time (cs)')

fig = D.iloc[0:500].plot(
    y=['Acceleration.forward', 'Acceleration.side', 'Acceleration.up'],
    ylabel='Acceleration $\\cdot 10 m/s^2$',
    xlabel='time in centiseconds'
)

# plt.show()


# 2.3. Scatterplot
# Make scatterplot of two columns Latitude and Longitude of the data between rows 0:500, and store the
# figure as fig_scatter. Make sure that Latitude is in the x-axis.

fig_scatter = D.iloc[0:500].plot(kind="scatter", x='Latitude', y='Longitude')
plt.ylabel('Longitude')
plt.xlabel('Latitude')

plt.show()
