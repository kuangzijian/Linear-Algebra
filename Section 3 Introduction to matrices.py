import numpy as np

# A zoo of matrices


# square vs. rectangular
S = np.random.randn(5,5)
R = np.random.randn(5,2) # 5 rows, 2 columns
print(R)

# identity
I = np.eye(3)
print(I)

# zeros
Z = np.zeros(4)
print(Z)

# diagonal
D = np.diag([ 1, 2, 3, 5, 2 ])
print(D)

# create triangular matrix from full matrices
S = np.random.randn(5,5)
U = np.triu(S)
L = np.tril(S)
print(L)

# concatenate matrices (sizes must match!)
A = np.random.randn(2,3)
B = np.random.randn(2,5)
C = np.concatenate((A,B), axis=1)
print(C)

# Matrix addition and subtraction

# create random matrices
A = np.random.randn(5,4)
B = np.random.randn(5,4)
C = np.random.randn(5,4)

# try to add them
A+B
A+C

# "shifting" a matrix
l = .3 # lambda
N = 5  # size of square matrix
D = np.random.randn(N,N) # can only shift a square matrix

Ds = D + l*np.eye(N)
print(Ds)

# Matrix-scalar multiplication

# define matrix and scalar
M = np.array([ [1, 2], [2, 5] ])
s = 2

# pre- and post-multiplication is the same:
print( M*s )
print( s*M )

#code challenge: Code challenge: is matrix-scalar multiplication a linear operation?


A = np.random.randn(3,2)
B = np.random.randn(3,2)
s = np.random.randint(10)
print(A,B,s)

r1 = s * (A + B)
print(r1)

r2 = s * A + s * B
print(r2)

# Diagonal and trace

M = np.round( 5*np.random.randn(4,4) )

# extract the diagonals
d = np.diag(M)

# notice the two ways of using the diag function
d = np.diag(M) # input is matrix, output is vector
D = np.diag(d) # input is vector, output is matrix
print(d)
print(D)

# trace as sum of diagonal elements
tr = np.trace(M)
tr2 = sum( np.diag(M) )
print(tr)
print(tr2)



#Complex matrix transpose

a = np.complex(3,2)
A = np.array([[a, 2, 3], [3,2,a]])

print(A)

B = np.transpose(A)

print(B)


# Code challenge: linearity of trace

A = np.round( 5* np.random.randn(4,4) )
B = np.round( 5* np.random.randn(4,4) )
print(A,B)

dA = np.diag(A)
dB = np.diag(B)
print(np.trace(A) + np.trace(B))
print(np.trace(A + B))

l = np.random.randint(10)
print(l)
print(np.trace(l*A))
print(l*np.trace(A))