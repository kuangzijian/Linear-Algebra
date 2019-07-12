import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# VIDEO: Algebraic and geometric interpretations
# 2-dimensional vector
v2 = [3, -2]

# 3-dimensional vector
v3 = [4, -3, 2]

# row to column (or vice-versa):
v3t = np.transpose(v3)


# plot them
plt.plot([0, v2[0]], [0, v2[1]])
plt.axis('equal')
plt.plot([-4, 4], [0, 0], 'k--')
plt.plot([0, 0], [-4, 4], 'k--')
plt.grid()
plt.axis((-4, 4, -4, 4))
plt.show()


# plot the 3D vector
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot([0, v3[0]], [0, v3[1]], [0, v3[2]])

# make the plot look nicer

ax.plot([0, 0], [0, 0], [-4, 4], 'k--')
ax.plot([0, 0], [-4, 4], [0, 0], 'k--')
ax.plot([-4, 4], [0, 0], [0, 0], 'k--')
plt.show()


# VIDEO: Vector addition/subtraction

# two vectors in R2
v1 = np.array([ 3, -1 ])
v2 = np.array([ 2,  4 ])

v3 = v1 + v2


# plot them
plt.plot([0, v1[0]],[0, v1[1]],'b',label='v1')
plt.plot([0, v2[0]]+v1[0],[0, v2[1]]+v1[1],'r',label='v2')
plt.plot([0, v3[0]],[0, v3[1]],'k',label='v1+v2')

plt.legend()
plt.axis('square')
plt.axis((-6, 6, -6, 6 ))
plt.grid()
plt.show()


# VIDEO: Vector-scalar multiplication
# vector and scalar
v1 = np.array([3, -1])
l = -.3
v1m = v1* l  # scalar-modulated

# plot them
plt.plot([0, v1[0]], [0, v1[1]], 'b', label='v_1')
plt.plot([0, v1m[0]], [0, v1m[1]], 'r:', label='\lambda v_1')

plt.axis('square')
plt.axis((-3, 3, -3, 3))
plt.grid()
plt.show()

# VIDEO: Vector-vector multiplication: the dot product
## many ways to compute the dot product

v1 = np.array([1, 2, 3, 4, 5])
v2 = np.array([0, -4, -3, 6, 5])

# method 1
dp = sum(np.multiply(v1, v2))

# method 2
dp = np.dot(v1, v2)

# method 3
dp = np.matmul(v1, v2)

# method 4
dp = 0  # initialize

# loop over elements
for i in range(0, len(v1)):
    # multiply corresponding element and sum
    dp = dp + v1[i] * v2[i]

print(dp)

# VIDEO: Dot product properties: associative and distributive
## Distributive property

# create random vectors
n = 10
a = np.random.randn(n)
b = np.random.randn(n)
c = np.random.randn(n)

# the two results
res1 = np.dot(a, (b+c))
res2 = np.dot(a, b) + np.dot(a, c)

# compare them
print([res1, res2])

## Associative property

# create random vectors
n = 10
a = np.random.randn(n)
b = np.random.randn(n)
c = np.random.randn(n)

# the two results
res1 = np.dot(a, np.dot(b, c))
res2 = np.dot(np.dot(a, b), c)

# compare them
print(res1)
print(res2)


### special cases where distributive property works!
# 1) one vector is the zeros vector
# 2) a==b==c


# code challenge 1
import numpy as np


m1 = np.random.randn(4, 6)
m2 = np.random.randn(4, 6)

print(m1, m2)

dp = []
for i in range(0, 6):
    # multiply corresponding columns and sum
    v1 = m1[:, i]
    v2 = m2[:, i]
    print(str(v1) + ' transpose ' + str(v2))
    dp.append(np.dot(v1, v2))

print('result is ' + str(dp))

# VIDEO: Vector length

# a vector
v1 = np.array([1, 2, 3, 4, 5, 6, ])

# methods 1-4, just like with the regular dot product, e.g.:
vl = np.sqrt(sum(np.multiply(v1, v1)))

# method 5: take the norm
vl = np.linalg.norm(v1)

print(vl)

# VIDEO: The dot product from a geometric perspective


# two vectors
v1 = np.array([ 2,  4, -3 ])
v2 = np.array([ 0, -3, -3 ])

# compute the angle (radians) between two vectors
ang = np.arccos( np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)) )


# draw them
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot([0, v1[0]],[0, v1[1]],[0, v1[2]],'b')
ax.plot([0, v2[0]],[0, v2[1]],[0, v2[2]],'r')

plt.axis((-6, 6, -6, 6))
plt.title('Angle between vectors: %s rad.' %ang)
plt.show()

# code challenge

v1 = np.random.randn(3)
v2 = np.random.randn(3)

s1 = np.random.random(1)
s2 = np.random.random(1)

print(v1,v2,s1,s2)

pd1 = np.dot(v1, v2)
print(pd1)

pd2 = np.dot(s1*v1, s2*v2)
print(pd2)

# VIDEO: Vector Hadamard multiplication
# create vectors
w1 = [ 1, 3, 5 ]
w2 = [ 3, 4, 2 ]

w3 = np.multiply(w1,w2)
print(w3)

# VIDEO: Vector outer product

v1 = np.array([  1, 2, 3 ])
v2 = np.array([ -1, 0, 1 ])

# outer product
np.outer(v1,v2)

# terrible programming, but helps conceptually:
op = np.zeros((len(v1),len(v1)))
for i in range(0,len(v1)):
    for j in range(0,len(v2)):
        op[i,j] = v1[i] * v2[j]


# VIDEO: Vector cross product
# create vectors
v1  = [ -3,  2, 5 ]
v2  = [  4, -3, 0 ]

# Python's cross-product function
v3a = np.cross( v1,v2 )
print(v3a)

# "manual" method
v3b = [ [v1[1]*v2[2] - v1[2]*v2[1]],
        [v1[2]*v2[0] - v1[0]*v2[2]],
        [v1[0]*v2[1] - v1[1]*v2[0]] ]

fig = plt.figure()
ax = fig.gca(projection='3d')

# draw plane defined by span of v1 and v2
xx, yy = np.meshgrid(np.linspace(-10,10,10),np.linspace(-10,10,10))
z1 = (-v3a[0]*xx - v3a[1]*yy)/v3a[2]
ax.plot_surface(xx,yy,z1)

## plot the two vectors
ax.plot([0, v1[0]],[0, v1[1]],[0, v1[2]],'k')
ax.plot([0, v2[0]],[0, v2[1]],[0, v2[2]],'k')
ax.plot([0, v3a[0]],[0, v3a[1]],[0, v3a[2]],'r')


ax.view_init(azim=150,elev=45)
plt.show()


# VIDEO: Hermitian transpose (a.k.a. conjugate transpose)

# create a complex number
z = np.complex(3,4)

# magnitude
print( np.linalg.norm(z) )

# by transpose?
print( np.transpose(z)*z )

# by Hermitian transpose
print( np.transpose(z.conjugate())*z )


# complex vector
v = np.array( [ 3, 4j, 5+2j, np.complex(2,-5) ] )
print( v.T )
print( np.transpose(v) )
print( np.transpose(v.conjugate()) )

# VIDEO: Unit vector

# vector
v1 = np.array([ -3, 6 ])

# mu
mu = 1/np.linalg.norm(v1)

v1n = v1*mu

# plot them
plt.plot([0, v1[0]],[0, v1[1]],'b',label='v1')
h=plt.plot([0, v1n[0]],[0, v1n[1]],'r',label='v1-norm')
plt.setp(h,linewidth=5)

# axis square
plt.axis('square')
plt.axis(( -6, 6, -6, 6 ))
plt.grid()
plt.legend()
plt.show()


# code challenge of unit vector

import numpy as np

v1 = np.random.randint(10, size=4)
v2 = np.random.randint(10, size=4)

print('v1: ' + str(v1))
print('v2: ' + str(v2))

lv1 = np.linalg.norm(v1)
lv2 = np.linalg.norm(v2)

print(lv1, lv2)

dp = np.dot(v1, v2)

print(dp)

nv1 = v1/lv1

nv2 = v2/lv2
print(np.linalg.norm(nv1), np.linalg.norm(nv2))
dp2 = np.dot(nv1, nv2)
print(dp2)


# VIDEO: Span
# set S
S1 = np.array([1, 1, 0])
S2 = np.array([1, 7, 0])

# vectors v and w
v = np.array([1, 2, 0])
w = np.array([3, 2, 1])

# draw vectors
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot([0, S1[0]],[0, S1[1]],[0, S1[2]],'r')
ax.plot([0, S2[0]],[0, S2[1]],[0, S2[2]],'r')

ax.plot([0, v[0]],[0, v[1]],[0, v[2]],'k')
ax.plot([0, w[0]],[0, w[1]],[0, w[2]],'b')

# now draw plane
xx, yy = np.meshgrid(range(30), range(30))
cp = np.cross(S1,S2)
z1 = (-cp[0]*xx - cp[1]*yy)*1./cp[2]
ax.plot_surface(xx,yy,z1)

plt.show()