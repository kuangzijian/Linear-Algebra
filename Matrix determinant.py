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

import numpy as np
import matplotlib.pyplot as plt

#Code challenge: Code challenge: large matrices with row exchanges

#Generate a 6x6 matrix;

M = np.round(np.random.rand(6,6) * 10)
print(M)
# -compute the determinant
print('rank is ', np.linalg.matrix_rank(M))
print('determinant is ', np.linalg.det(M))

# -swap one row, det again
print('swap first row with second row')
M1 = M[[1,0,2,3,4,5]]
print('new M is\n', M1)
print('new determinant is ', np.linalg.det(M1))
# -swap two rows
print('swap first row with second row')
M2 = M[[1,2,0,3,4,5]]
print('new M is\n', M2)
print('new determinant is ', np.linalg.det(M2))

#Code challenge: determinant of shifted matrices
# (as shift the matrix further and further towards the identity matrix,
# the abs of determinant is larger and larger)

#Generate a 6x6 matrix;
# shift amount (l=lambda)
lamdas = np.linspace(0, 0.1, 30)
dets = []
for deti in range (0, len(lamdas)):
    for i in range(0, 999):
        M = np.round(np.random.rand(20,20) * 10)

    #impose a linear dependence
        M[:,0] = M[:,1]

    #shift the matrix (0->0.1 times the identity matrix) (lambda)
        M = M + lamdas[deti]*np.eye(20,20)

    #comput the abs(determinant) of the shifted matrix
        det = []
        det.append(np.abs(np.linalg.det(M)))
    dets.append(np.mean(det))
#repeat 1000 times, take the averge abs(det)
#plot of det as a function of lambda
plt.plot(lamdas, dets, 's')
plt.show()
