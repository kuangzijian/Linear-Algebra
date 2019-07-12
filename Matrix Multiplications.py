import numpy as np
import matplotlib.pyplot as plt
import math

# VIDEO: Standard matrix multiplication, parts 1 & 2
## rules for multiplication validity

m = 4
n = 3
k = 6

# make some matrices
A = np.random.randn(m,n)
B = np.random.randn(n,k)
C = np.random.randn(m,k)

# test which multiplications are valid.
# Think of your answer first, then test.
np.matmul(A,B)
np.matmul(A,A)
np.matmul(np.matrix.transpose(A),C)
np.matmul(B,np.matrix.transpose(B))
np.matmul(np.matrix.transpose(B),B)
np.matmul(B,C)
np.matmul(C,B)
np.matmul(np.matrix.transpose(C),B)
np.matmul(C,np.matrix.transpose(B))


#Code challenge: matrix multiplication by layering


A = np.abs(np.round(5*np.random.randn(4,2)))
B = np.abs(np.round(5*np.random.randn(2,3)))

print(A)
print(B)


r1 = 0
for i in range(0, len(B)):
    r1 = r1 + np.outer(A[:,i], B[i])
    print(A[:,i])
    print(B[i])

print(r1)

print(np.matmul(A, B))
