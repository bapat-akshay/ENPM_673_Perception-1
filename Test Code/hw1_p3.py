import numpy as np

# First entry in list set as None for more intuitive indexing
x = [None, 5, 150, 150, 5]
y = [None, 5, 5, 150, 150]
xp = [None, 100, 200, 220, 100]
yp = [None, 100, 80, 80, 200]

A = np.array([[-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]], 
	[0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]], 
	[-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]], 
	[0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]], 
	[-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]], 
	[0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]], 
	[-x[4], -y[4], -1, 0, 0, 0, x[4]*xp[4], y[4]*xp[4], xp[4]], 
	[0, 0, 0, -x[4], -y[4], -1, x[4]*yp[4], y[4]*yp[4], yp[4]]])


A_tp = np.transpose(A)
AAT = np.matmul(A, A_tp)
ATA = np.matmul(A_tp, A)

L2, V = np.linalg.eig(ATA)

sigma = np.zeros((9,9))
for i in range(9):
	sigma[i][i] = np.sqrt(abs(L2[i]))

U = np.matmul(np.matmul(A, V), np.linalg.inv(sigma))
V_tp = np.transpose(V)

ans = np.matmul(np.matmul(U, sigma), V_tp)

print(ans)
print(A)