import numpy as np 
import math
from numpy import linalg as LA

c = np.array([
(5,5),
(-1,7)
])

c_transpose = np.transpose(c)

# intermediate_matrix = c_transpose * c
# intermediate_matrix = np.multiply(c_transpose,c)   #does the dot product
intermediate_matrix = np.matmul(c_transpose,c)

# print(f'C transpose:\n {c_transpose}')

# print(f'Intermediate matrix:\n {intermediate_matrix}')

# eigen_value_matrix = np.diag((1, 2, 3))
# print(diagonal_test)

w, v = LA.eig(intermediate_matrix)
# print(w) #Eigen values
# print(v) #Eigen vector matrix

V_eigen_vector_matrix = v
V_transpose = np.transpose(V_eigen_vector_matrix)

eigen_square_root = []
# print(w[1])

for i in range(0,len(w)):
	eigen_square_root.append(math.sqrt(w[i]))

# print(eigen_square_root)

Sigma_eigen_value_matrix = np.diag(eigen_square_root)

# print(Sigma_eigen_value_matrix)
# print(math.sqrt(4)) 

Sigma_inverse = np.linalg.inv(Sigma_eigen_value_matrix)

U_matrix = np.matmul(c, np.matmul(V_eigen_vector_matrix,Sigma_inverse))

print(f'SVD decomposition of C is:\nU: \n {U_matrix}')	
print(f'Sigma :\n {Sigma_eigen_value_matrix}')	
print(f'V transpose :\n {V_transpose}')

SVD_check = np.matmul(U_matrix, np.matmul(Sigma_eigen_value_matrix, V_transpose))
print(SVD_check)















