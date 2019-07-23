import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import *
#Standard matrix multiplication, parts 1 & 2
## rules for multiplication validity

m = 4
n = 3
k = 6

# make some matrices
A = np.random.randn(m,n)
B = np.random.randn(n,k)
C = np.random.randn(m,k)

# test which multiplications are valid.
np.matmul(A,B)
#np.matmul(A,A)
np.matmul(np.matrix.transpose(A),C)
np.matmul(B,np.matrix.transpose(B))
np.matmul(np.matrix.transpose(B),B)
#np.matmul(B,C)
#np.matmul(C,B)
#np.matmul(np.matrix.transpose(C),B)
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

#Order-of-operations on matrices
n = 2
L = np.random.randn(n,n)
I = np.random.randn(n,n)
V = np.random.randn(n,n)
E = np.random.randn(n,n)

# result of "forward" multiplication and then transpose
res1 = np.matrix.transpose( L @ I @ V @ E )

# result of "flipped" multiplication of transposed matrices
res2 = np.matrix.transpose(E) @ np.matrix.transpose(V) @ np.matrix.transpose(I) @ np.matrix.transpose(L)

# test equality by subtracting (ignore possible computer rounding errors)
res1-res2

#Matrix-vector multiplication
# number of elements
m = 4

# create matrices
N = np.round( 10*np.random.randn(m,m) )
S = np.round( np.matrix.transpose(N)*N/m**2 ) # scaled symmetric

# and vector
w = np.array([-1, 0, 1, 2])
print(S)
print(w)
print(N)

print("with symmetric matrix")
# NOTE: The @ symbol for matrix multiplication is relatively new to Python, a@b is the same as numpy.dot or a.dot(b)
print(S@w)    # 1
print(np.matrix.transpose(S@w)) # 2
print(w@S)    # 3
print(np.matrix.transpose(w)@np.matrix.transpose(S))  # 4
print(np.matrix.transpose(w)@S)   # 5

print("with nonsymmetric matrix")
print(N@w)    # 1
print(np.matrix.transpose(N@w)) # 2
print(w@N)    # 3
print(np.matrix.transpose(w)@np.matrix.transpose(N))  # 4
print(np.matrix.transpose(w)@N)   # 5

#2D transformation matrices

# 2D input vector
v = np.array([ 3, -2 ])

# 2x2 transformation matrix
A = np.array([ [1,-1], [2,1] ])

# output vector is Av (convert v to column)
w = A@np.matrix.transpose(v)


# plot them
plt.plot([0,v[0]],[0,v[1]],label='v')
plt.plot([0,w[0]],[0,w[1]],label='Av')

plt.grid()
plt.axis((-6, 6, -6, 6))
plt.legend()
plt.title('Rotation + stretching')
plt.show()

## pure rotation

# 2D input vector
v = np.array([ 3, -2 ])

# 2x2 rotation matrix
th = np.pi/30
A = np.array([ [math.cos(th),-math.sin(th)], [math.sin(th),math.cos(th)] ])

# output vector is Av (convert v to column)
w = A@np.matrix.transpose(v)


# plot them
plt.plot([0,v[0]],[0,v[1]],label='v')
plt.plot([0,w[0]],[0,w[1]],label='Av')

plt.grid()
plt.axis((-4, 4, -4, 4))
plt.legend()
plt.title('Pure rotation')
plt.show()

#code challenge: Pure and impure rotation matrices

v = np.array([ 3, -2 ])

# 2x2 rotation matrix
ths = np.linspace(0, 2*np.pi,100)

vecmags = np.zeros([len(ths),2])

for i in range(0, len(ths)):
    th = ths[i]
#inpure transformation matrix
    A1 = np.array([ [2*math.cos(th),-math.sin(th)], [math.sin(th),math.cos(th)] ])

#pure transformation matrix
    A2 = np.array([ [math.cos(th),-math.sin(th)], [math.sin(th),math.cos(th)] ])

# output vector is Av (convert v to column)
    vecmags[i, 0] = np.linalg.norm(A1 @ v)
    vecmags[i, 1] = np.linalg.norm(A2 @ v)

# plot them
plt.plot(ths,vecmags)

plt.grid()

plt.legend(["inpure transformation","pure transformation matrix"])
plt.title('Pure and impure rotation matrices')
plt.show()


#Additive and multiplicative matrix identities
# size of matrices
n = 4

A = np.round( 10*np.random.randn(n,n) )
I = np.eye(n,n)
Z = np.zeros((n,n))

# test both identities
np.array_equal( A@I , A   )
np.array_equal( A   , A@I )
np.array_equal( A   , A+I )

np.array_equal( A   , A+I )
np.array_equal( A+Z , A@I )

#Additive and multiplicative symmetric matrices

## the additive method

# specify sizes
m = 5
n = 5

# create matrices
A = np.random.randn(m,n)
S = ( A + np.matrix.transpose(A) )/2

# A symmetric matrix minus its transpose should be all zeros
print( S-np.matrix.transpose(S) )

## the multiplicative method

# specify sizes
m = 5
n = 3

# create matrices
A   = np.random.randn(m,n)
AtA = np.matrix.transpose(A)@A
AAt = A@np.matrix.transpose(A)

# first, show that they are square
print( AtA.shape )
print( AAt.shape )


# next, show that they are symmetric
print( AtA - np.matrix.transpose(AtA) )
print( AAt - np.matrix.transpose(AAt) )


#Element-wise (Hadamard) multiplication
# any matrix sizes
m = 13
n =  2

# ...but the two matrices must be the same size
A = np.random.randn(m,n)
B = np.random.randn(m,n)

# note the different syntax compared to @ for matrix multiplication
C = np.multiply( A,B )

print(C)

#code challenge: Symmetry of combined symmetric matrices

print("Create two symmetric matrices")
S = np.round( 2*np.random.randn(3,2) )
S1 = S.dot(np.transpose(S))
print(S1)
S = np.round( 2*np.random.randn(3,2) )
S2 = S.dot(np.transpose(S))
print(S2)

print("compute sum, multiplication, and Hadamard multiplication of the two matrices")
#determine whether the result is still symmetric
print(S1+S2)
print(S1.dot(S2))
print(S1*S2)

#Multiplication of two symmetric matrices

a,b,c,d,e,f,g,h,k,l,m,n,o,p,q,r,s,t,u = symbols('a b c d e f g h k l m n o p q r s t u', real=True)

# symmetric and constant-diagonal matrices
A = Matrix([ [a,b,c,d],
             [b,a,e,f],
             [c,e,a,h],
             [d,f,h,a]   ])

B = Matrix([ [l,m,n,o],
             [m,l,q,r],
             [n,q,l,t],
             [o,r,t,l]   ])


# confirmation that A and B are symmetric
print( A - A.transpose() )
print( B - B.transpose() )

# ... and constant diagonal
for i in range(0,np.size(A,0)):
    print( A[i,i] )
for i in range(0,np.size(B,0)):
    print( B[i,i] )

# but AB neq (AB)'
A@B - (A@B).T

# maybe for a submatrix?
n = 3
A1 = A[ 0:n,0:n ]
B1 = B[ 0:n,0:n ]

A1@B1 - (A1*B1).T

#Frobenius dot-product
# any matrix sizes
m = 9
n = 4

# but the two matrices must be the same size
A = np.random.randn(m,n)
B = np.random.randn(m,n)

# first vectorize, then vector-dot-product
Av = np.reshape( A,m*n, order='F' ) # order='F' reshapes by columns instead of by rows
Bv = np.reshape( B,m*n, order='F' )
frob_dp = np.dot( Av,Bv )

# trace method
frob_dp2 = np.trace( np.matrix.transpose(A)@B )
print(frob_dp2)
print(frob_dp)

# matrix norm
Anorm  = np.linalg.norm(A,'fro')
Anorm2 = np.sqrt( np.trace( np.matrix.transpose(A)@A ) )
print(Anorm)
print(Anorm2)


#Code challenge: standard and Hadamard multiplication for diagonal matrices

#Create two matrices 4x4 full and diagonal

D1 = np.random.randn(4,4)
D2 = np.diag([4,5,6,7])

#multiply each matrix by itself (A*A): standard and hadmard multiplications

RS1 = D1.dot(D1)
RS2 = D2.dot(D2)
RH1 = D1*D1
RH2 = D2*D2

print(D1)
print(RS1)
print(RH1)
print(D2)
print(RS2)
print(RH2)


