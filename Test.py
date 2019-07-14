import numpy as np
import matplotlib.pyplot as plt

#Code challenge: is this vector in the span of this set?

v =np.array([[1,2,3,4]])
print(v)

S =(np.array([[4,3,6,2],[0,4,0,1]]))
Sv = np.concatenate((S,v), axis=0)
print(S)
print(np.linalg.matrix_rank(S))
print(Sv)
print(np.linalg.matrix_rank(Sv))

T = (np.array([[1,2,2,2],[0,0,1,2]]))
Tv = np.concatenate((T,v), axis=0)

print(T)
print(np.linalg.matrix_rank(T))
print(Tv)
print(np.linalg.matrix_rank(Tv))