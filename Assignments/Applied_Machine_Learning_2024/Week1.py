import numpy as np

# 1
x = 3
print(x)

l = list((3, 4, 5))
print(l)

a = np.array([1, 2, 3])
print(a)

print(" ")
# 2. Use numpy functions for creating arrays
# 2.1 Create a numpy array, values, which has 100 consequent values between 0 and pi.
# using np.linspace().Use np.pi as constant for pi.
# 2.2 Create a two-dimensional array M of 5x5 zeros or 5x5 ones. using appropriate functions
# 2.3 Create one more 2-dimensional array with the following code A=np.array([[1,2,3], [4,5,6]])

values = np.linspace(0, stop=np.pi)
print(values)

M = np.ones((5, 5))
print(M)

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A)

# Study the shape of previously created numpy arrays by using function .shape() included in the arrays you created.
# Try to calculate also the sum of all values in those arrays with .sum() function. Store the sum of array values to
# variable called s.
# Transpose a two-dimensional array with function .T for example A.T and name the result as tA
# Transposing a one dimensional array looks sadly slightly illogical, because it needs to be also converted to
# two-dimensional at the same time. The row vector values can be transposed to column vector with syntax
# values.reshape(-1,1). Try it and name the result as tvalues.
# Try also to transpose the column vector back to row vector

shape = np.shape(values)
print(shape)

s = values.sum()
print(s)

tA = A.T
print(tA)

tvalues = values.reshape(-1, 1)
print(tvalues)


# Create a function squares, which takes a an integer, n, as a parameter, and prints the values and squares of n
# first integers, and returns the square of n as its value.

def squares(n):
    for i in range(n):
        print(i, i ** 2)
    return n ** 2


squares(10)
