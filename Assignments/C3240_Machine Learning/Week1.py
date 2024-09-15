import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Student Task A1.1
# Create a numpy array x that represents the vector 𝐱=(5,4,3)𝑇
# and another numpy array y that represents the vector 𝐲=(4,3,2)𝑇
# Complete the function sum_matrix which should read in two numpy arrays of the same shape. The function should return
# a numpy array with the same shape of the inputs and whose entries are sums of the corresponding entries in the two input arrays.
# Similar to sum_matrix, complete the function product_matrix that returns a numpy array whose entries are products of
# the entries of the input numpy arrays.
# NOTE: In this exercise we equate 1-dimensional numpy arrays with column vectors, DO NOT create arrays with e.g. shape (1,3)

x = np.array([5, 4, 4])
y = np.array([4, 3, 2])


def sum_matrix(x, y):
    if x.shape == y.shape:
        return np.add(x, y)
    else:
        raise ValueError("Input arrays must have the same shape.")


result = sum_matrix(x, y)
print(result)


def product_matrix(x, y):
    if x.shape == y.shape:
        return np.multiply(x, y)
    else:
        raise ValueError("Input arrays must have the same shape.")


result = product_matrix(x, y)
print(result)

# Student Task A1.2

# Create a numpy array A of shape (3, 2) that represents the 3×2 integer matrix
# Complete the three functions in the code snippet:
# first_row that should return a 1-D numpy array that represents the first row of the matrix corresponding to the input array.
# second_column that should return 1-D numpy array ... second column ...
# second_row_and_column that should return a single number which is contained in the second row and second column of the input matrix.

A = np.array([[1, 2], [3, 4], [5, 6]])


# B = np.random.randint(0, 10, (3, 2))
# C = np.ones((3, 2))
# print(A, B, C)


def first_row(A):
    return A[0, :]


def second_column(A):
    return A[:, 1]


def second_row_and_column(A):
    return A[1, 1]


print(first_row(A), second_column(A), second_row_and_column(A))

# Create a numpy array B of shape (2,2) that represents the matrix [[1,5],[3,7]]
# Create a numpy array C of shape (2,2) that represents the matrix [[2,6],[4,8]]
# Complete the function matrix_mult which reads in numpy arrays B and C and returns a numpy array that represents
# The matrix multiplication of the matrices represented by B and C.

B = np.array([[1, 5], [3, 7]])
C = np.array([[2, 6], [4, 8]])


def matrix_mult(B, C):
    return np.dot(B, C)


print(matrix_mult(B, C))

# Student Task A1.4
# Consider the code line A=np.array([[1, 0], [0, 1], [1, 1]]) which creates a numpy array A. What is the shape of the numpy array?
# (Set the variable Answer to the index of the correct answer)
# Answer 1: the shape is (3, 2).
# Answer 2: the shape is (2, 1).

numpy_array_A = np.array([[1, 0], [0, 1], [1, 1]])
shape = numpy_array_A.shape
print(shape)  # => (3,2)

# Reshape demo
# create a 1-D numpy array
P = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# print('P:\n', P)
# print('The shape of P is: ', P.shape)

# reshape P to a 2-D array, the size of the second dimension is 1, the first dimension is inferred
# P_1 = P.reshape((-1, 1))
# print('\nP_1:\n', P_1)
# print('The shape of P_1 is: ', P_1.shape)

# reshape P to a 2-D array with the shape (4, 2)
# P_2 = P.reshape((4, 2))
# print('\nP_2:\n', P_2)
# print('The shape of P_2 is: ', P_2.shape)


# Student Task A1.5
# Your task is to reshape P to a new ndarray P_test with the shape (2, 4)

P_test = P.reshape((2, 4))
print('\nP_test:\n', P_test)
print('The shape of P_test  is: ', P_test.shape)

# Student Task A1.6

# Consider the weather observations recorded in air_temp.csv and loaded into the dataframe data. We define a datapoint to represent an entire day,

#    First datapoint represents the day 2021-06-01,
#    Second datapoint represents the day 2021-06-02,
#    Third datapoint represents the day 2021-06-03,
#   ...

# The total number 𝑚 of datapoints is the number of days for which data contains weather recordings for daytime 11:00 and 12:00.
# We characterize the 𝑖-th datapoint (day) using
# - the temperature recorded at 11:00 during the 𝑖th day as its feature 𝑥(𝑖)
# - the temperature recorded at 12:00 during the 𝑖th day as its label 𝑦(𝑖)
# Store the feature values in a numpy array X of shape (m,1) and the label values in a numpy array y of shape (m,).

df = pd.read_csv('air_temp.csv')

X = df[df["Time"] == "01:00"]['Air temperature (degC)'].values.reshape((-1, 1))
Y = df[df["Time"] == "00:00"]['Air temperature (degC)'].values
print(X, Y)
