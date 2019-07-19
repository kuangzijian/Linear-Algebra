import numpy as np

# Code challenge: determinant of small and large singular matrices

# singular matrix (reduced-rank matrix) has a determinant of 0

S = np.random.randn(3,3)
S[:,2] = S[:,1]
print(S)
DetS = np.linalg.det(S)
print('S determinant is ', DetS)

# generate a 2x2 matrix of integers, and with linear dependencies
# compute the rank

A = np.round(np.random.randn(3,3) * 10)
A[:,2] = A[:,1]
print(A)
print('A rank is ', np.linalg.matrix_rank(A))
print('A determinant is ', np.linalg.det(A))

#generate mxm matrices, impose linear dependencies

m = 3

M = np.random.randn(m,m)
M[:,m-1] = M[:,m-2]
print(M)
print('M rank is ', np.linalg.matrix_rank(A))
print('M determinant is ', np.linalg.det(A))

m = 100

M = np.random.randn(m,m)
M[:,m-1] = M[:,m-2]
print(M)
print('larger M rank is ', np.linalg.matrix_rank(A))
print('larger M determinant is ', np.linalg.det(A))
