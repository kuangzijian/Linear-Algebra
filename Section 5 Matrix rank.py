import numpy as np
import matplotlib.pyplot as plt
import math


# Computing rank: theory and practice
# make a matrix
m = 4
n = 6

# create a random matrix
A = np.random.randn(m,n)

# what is the largest possible rank?
ra = np.linalg.matrix_rank(A)
print('rank=' + str(ra))


# set last column to be repeat of penultimate column
B = A
B[:,n-1] = B[:,n-2]
rb = np.linalg.matrix_rank(B)
print('rank=' + str(rb))

## adding noise to a rank-deficient matrix

# square for convenience
A = np.round( 10*np.random.randn(m,m) )

# reduce the rank
A[:,m-1] = A[:,m-2]

# noise level
noiseamp = .001

# add the noise
B = A + noiseamp*np.random.randn(m,m)

print('rank (w/o noise) = ' + str(np.linalg.matrix_rank(A)))
print('rank (with noise) = ' + str(np.linalg.matrix_rank(B)))

#Code challenge: reduced-rank matrix via multiplication

#create a 10x10 matrix with rank 4 (use matrix multiplication)
#generalize the procedure to create any MxN matrix with Rank r

m = 10
n = 10
r = 4

A = np.random.randn(m,r)
B = np.random.randn(r,n)
print(np.linalg.matrix_rank(A.dot(B)))
print(np.shape(A.dot(B)))

#Code challenge: scalar multiplication and rank

#test whether the matrix rank is invariant to scalar multiplication

m = 5
n = 4
r = 3

s = 2
F = np.random.randn(m,n)
R = np.random.randn(m,r).dot(np.random.randn(r,n))

print("Full rank matrix:")
print(np.linalg.matrix_rank(F))
print(np.linalg.matrix_rank(F*s))
print(s*np.linalg.matrix_rank(F))

print("Low rank matrix:")
print(np.linalg.matrix_rank(R))
print(np.linalg.matrix_rank(R*s))
print(s*np.linalg.matrix_rank(R))

# Rank of A^TA and AA^T

# matrix sizes
m = 14
n =  3

# create matrices
A = np.round( 10*np.random.randn(m,n) )

AtA = np.matrix.transpose(A)@A
AAt = A@np.matrix.transpose(A)

# get matrix sizes
sizeAtA = AtA.shape
sizeAAt = AAt.shape

# print info!
print('AtA: %dx%d, rank=%d' %(sizeAtA[0],sizeAtA[1],np.linalg.matrix_rank(AtA)))
print('AAt: %dx%d, rank=%d' %(sizeAAt[0],sizeAAt[1],np.linalg.matrix_rank(AAt)))


#Code challenge: rank of multiplied and summed matrices

#rank of AB <= min(rank(A),rank(B)) ;  rank of A+B <= rank(A) + rank(B)
m = 2
n = 5

# create matrices

A = np.random.randn(2,5)
B = np.random.randn(2,5)
print(np.linalg.matrix_rank(A))
print(np.linalg.matrix_rank(B))

AtA = np.transpose(A).dot(A)
BtB = np.transpose(B).dot(B)

print("rank of AB <= min(rank(A),rank(B))")
print(np.linalg.matrix_rank(AtA))
print(np.linalg.matrix_rank(BtB))
print(np.linalg.matrix_rank(AtA.dot(BtB)))

print("rank of A+B <= rank(A) + rank(B)")
print(np.linalg.matrix_rank(AtA))
print(np.linalg.matrix_rank(BtB))
print(np.linalg.matrix_rank(AtA) + np.linalg.matrix_rank(BtB))


#Making a matrix full-rank by "shifting"

# size of matrix
m = 30

# create the square symmetric matrix
A = np.random.randn(m,m)
A = np.round( 10*np.matrix.transpose(A)@A )

# reduce the rank
A[:,0] = A[:,1]

# shift amount (l=lambda)
l = .01

# new matrix
B = A + l*np.eye(m,m)

# print information
print('rank(w/o shift) = %d' %np.linalg.matrix_rank(A))
print('rank(with shift) = %d' %np.linalg.matrix_rank(B))
