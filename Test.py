import numpy as np
import matplotlib.pyplot as plt

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

