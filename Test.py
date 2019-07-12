import numpy as np

A = np.abs(np.round(5*np.random.randn(4,2)))
B = np.abs(np.round(5*np.random.randn(2,3)))

print(A)
print(B)
print("Test")

r1 = 0
for i in range(0, len(B)):
    r1 = r1 + np.outer(A[:,i], B[i])
    print(A[:,i])
    print(B[i])

print(r1)

print(np.matmul(A, B))