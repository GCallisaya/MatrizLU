import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve

# Ejemplo de matriz
A = np.array([[8, 4, -1],
              [-2, 5, 1],
              [2, -1, 6]])

# Vector de constantes (Ax = b)
b = np.array([11, 4, 7])

# Descomposición LU
P, L, U = lu(A)

print("Matriz P:")
print(P)
print("Matriz L:")
print(L)
print("Matriz U:")
print(U)

# Solución del sistema Ax = b usando la descomposición LU
lu, piv = lu_factor(A)
x = lu_solve((lu, piv), b)

print("\nSolución del sistema Ax = b:")
print(x)
