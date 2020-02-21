# ENPM 673 - Perception for autonomous robots

# Python code to compute the Singular Value Decomposition for Matrix A

import numpy as np 
import math
from numpy import linalg as LA

# First entry in list set as None for more intuitive indexing
x = [None, 5, 150, 150, 5]
y = [None, 5, 5, 150, 150]
xp = [None, 100, 200, 220, 100]
yp = [None, 100, 80, 80, 200]

# Defining matrix A
A = np.array([[-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]], 
	[0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]], 
	[-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]], 
	[0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]], 
	[-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]], 
	[0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]], 
	[-x[4], -y[4], -1, 0, 0, 0, x[4]*xp[4], y[4]*xp[4], xp[4]], 
	[0, 0, 0, -x[4], -y[4], -1, x[4]*yp[4], y[4]*yp[4], yp[4]]])

print("The matrix A for which SVD needs to be done is:-")
print(A)

A_transpose = np.transpose(A)
intermediate_matrix = np.matmul(A_transpose,A)

#compute eigen vectors and eigen values
w, v = LA.eig(intermediate_matrix)
# print(w) #w is the list for Eigen values
# print(v) #v is the Eigen vector matrix where columns of V correspond to 
# eigen vectors of (A_transpose*A)

#Define the matrix V
V_eigen_vector_matrix = v
V_transpose = np.transpose(V_eigen_vector_matrix)

# Make an empty list called eigen_square_root
eigen_square_root = []

#take the square root of eigen values
for i in range(0,len(w)):
	eigen_square_root.append(math.sqrt(abs(w[i])))

#Generate eigen value matrix
Sigma_eigen_value_matrix = np.diag(eigen_square_root)

#Find inverse of eigen value matrix
Sigma_inverse = np.linalg.inv(Sigma_eigen_value_matrix)

# Find the U matrix
U_matrix = np.matmul(A, np.matmul(V_eigen_vector_matrix,Sigma_inverse))

#Print all the matrices after SVD decomposition
print(f'SVD decomposition of A is:\nU: \n {U_matrix}')	
print(f'Sigma :\n {Sigma_eigen_value_matrix}')	
print(f'V transpose :\n {V_transpose}')

# Uncomment the following section to multiply U, sigma and V_tranpose 
# You'll get the original matrix A again which shows SVD is correct
# SVD_check = np.matmul(U_matrix, np.matmul(Sigma_eigen_value_matrix, V_transpose))
# print(f'SVD check matrix:\n {SVD_check}')
		
print('Homography matrix is:- \n')
last_column_v = V_eigen_vector_matrix[:,8]

homography_matrix = np.reshape(last_column_v, (3,3))
print(homography_matrix)














