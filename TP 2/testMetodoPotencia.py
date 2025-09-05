import numpy as np
import os
import matplotlib.pyplot as plt

# Creo una matriz random


def test_metodo_potencia(X):
	n = len(X)
	# Calculo los autovalores y autovectores con numpy

	autovalores, autovectores = np.linalg.eig(X)
	idx = np.argsort(autovalores)
	idx = idx[::-1]    

	#Ordenar los autovectores
	vectors = autovectores[:,idx]
	values = autovalores[idx]
	#Escribo la matriz de covarianza en un archivo de texto para que la funcion de c++ la pueda leer
	
	f = open(str(n)  + ".txt", "w")
	f.write(str(n))
	f.write (" ")
	f.write(str(n))
	f.write("\n")
	for i in range(n):
			for j in range(n):
				f.write(str(X[i][j]))		# Queremos los autovectores de la matriz de covarianza
				f.write (" ")
			f.write("\n")
	f.close()

	# Ejecuto el programa de c++ que calcula el metodo de la potencia
	n = str(n)

	os.system("./metodoPotencia " + n + " " + n )
	
	eigenvalues = np.loadtxt(n + "_autovalores.txt")
	eigenvectors = np.loadtxt(n + "_autovectores.txt")

	equal = np.allclose(values, eigenvalues, rtol=1e-05, atol=1e-08, equal_nan=False) or np.allclose(values, -eigenvalues, rtol=1e-05, atol=1e-08, equal_nan=False)


	error = 0
	equal_vectors = True

	cos_sim_values = []  # create a list to store the cosine similarity values
	cos_sim_prom = 0
	for i in range(vectors.shape[1]):
		# Comparo los autovectores con los de numpy por  cosine similarity
		cos_sim = np.dot(eigenvectors[:, i], vectors[:, i]) / (np.linalg.norm(eigenvectors[:, i]) * np.linalg.norm(vectors[:, i]))
		cos_sim_values.append(abs(cos_sim))  # add the cosine similarity value to the list
		cos_sim_prom += abs(cos_sim)
		if not np.isclose(abs(cos_sim), 1, rtol=1e-03, atol=1e-03, equal_nan=False):
			equal_vectors = False
			error += 1
			print(f'Vector {i} Cosine similarity between eigenvectors: {cos_sim}')
		else:
			print(f"Los autovectores son iguales para el autovector {i}.")
	if equal_vectors:
		print("Todos los autovectores son iguales.")
	print("cant error ", error)
	cos_sim_prom = cos_sim_prom / vectors.shape[1]
	print("cos_sim_prom ", cos_sim_prom)

	# plot the cosine similarity values
	plt.scatter(range(vectors.shape[1]), cos_sim_values)
	plt.xlabel('Eigenvector index')
	plt.ylabel('Cosine similarity')
	#plt.ylim([0, 2])  # set the y-axis limits to show values from 0 to 1
	plt.show()

	#Borro los archivos de texto
	os.system("rm " + n + ".txt")
	os.system("rm " + n + "_autovalores.txt")
	os.system("rm " + n + "_autovectores.txt")
	
	return values, vectors


#### Matriz diagonal con diferentes autovalores
diagonal_elements = np.array([1, 2, 3, 4,5,6,7,8])
X = np.diag(diagonal_elements)
test_metodo_potencia(X)
#Borrar el arcchivo de texto


#### Matriz simetrica

X = np.random.rand(10,10)
X = X @ X.T
test_metodo_potencia(X)

#### Matriz simettrica mas grande
X = np.random.rand(100,100)
X = X @ X.T
test_metodo_potencia(X)
"""

########## Que pasa con matrices que no cumplen la precondicion ############
# Creo una matriz que no cumple la precondicion
# Matriz cuyos autovalores sean iguales
diagonal_elements = np.array([10,9,10,8,1])
X = np.diag(diagonal_elements)
test_metodo_potencia(X)

### MATRIZ NO SIMETRICA
X = np.random.rand(3,3)
test_metodo_potencia(X)"""


