import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import multiprocessing
import sklearn.datasets as skd

def generate_diagonal_dominant_matrix(size):
    A = np.random.randint(-10, 11, (size, size))
    np.fill_diagonal(A, np.abs(A.sum(axis=1)) + np.random.randint(15, 20))
    return A

def generar_sdp(size):
    A = np.random.randint(1, 11, (size, size))
    np.fill_diagonal(A, np.abs(A.sum(axis=1)) + np.random.randint(1, 10))
    A = A @ A.T
    return A

def tJacobiConverge(dim, n = 0):
    if n == 600:
        return generate_diagonal_dominant_matrix(dim)
    # D-1 * (L + U)
    A =  np.random.rand(dim, dim)
    D = np.diag(np.diag(A))
    L = -np.tril(A, k=-1)
    U = -np.triu(A, k=1)
    D_inv = np.linalg.inv(D)
    LU = L + U
    B = D_inv @ LU
    # Calcular el radio espectral de B
    eigenvalues = np.linalg.eigvals(B)
    rho = np.max(np.abs(eigenvalues))
    if rho < 1:
        print("N  iteración: ", n)
        return A
    else:
        return tJacobiConverge(dim, n + 1)



def generar_testeo(dim, cant_matrices):
    
    jacobiNoConverge = []
    jacobiConverge = []
    gaussSeidelNoConverge = []
    gaussSeidelConverge = []
    jacobiGaussidelConvergen = []

    for i in range(cant_matrices):
        size = dim
        
        # Obtener la descomposición en valores singulares de la matriz
        A = tJacobiConverge(dim)
        x = np.random.randint(1, 11, (size, 1))
        
        #A = np.identity(dim)
        #x = np.random.rand(dim)
        #A = generar_sdp(dim)
        b = A @ x

        # f = open( "jacobi_matriz"  + ".txt", "w")
        with open('matriz.txt', 'w') as f:
            # Escribir las dimensiones en la primera línea
            f.write(f'{dim} {dim}\n')
            # Escribir la matriz en el resto del archivo
            np.savetxt(f, A)
        with open('solucion.txt', 'w') as f:
            # Escribir las dimensiones en la primera línea
            f.write(f'{dim}\n')
            # Escribir la matriz en el resto del archivo
            np.savetxt(f, b)
        with open('elX.txt', 'w') as f:
            # Escribir las dimensiones en la primera línea
            f.write(f'{dim}\n')
            # Escribir la matriz en el resto del archivo
            np.savetxt(f, x)
        os.system("./jacobi matriz.txt "  + "solucion.txt "+ "testeo " + "elX.txt") # O(m^4) siendo m la cantidad de pixeles de cada imagen
        converge = np.loadtxt("converge.txt")
        if converge[0]== 1 and converge[2] == 1 :
            jacobiGaussidelConvergen.append(A)
        if converge[0] == 0:
            jacobiNoConverge.append(A)
        else:
            jacobiConverge.append(A)
        if converge[3] == 0:
            gaussSeidelNoConverge.append(A)
        else:
            gaussSeidelConverge.append(A)
        
    np.save("jacobigausSeidelConvergen"+ str(dim), jacobiGaussidelConvergen)
    np.save("jacobiNoConverge"+ str(dim), jacobiNoConverge)
    np.save("jacobiConverge"+ str(dim), jacobiConverge)
    np.save("gaussSeidelNoConverge"+ str(dim), gaussSeidelNoConverge)
    np.save("gaussSeidelConverge"+ str(dim), gaussSeidelConverge)



def generar_convergen_jacobi_gauss(dim, cantmatrices):
    
    jacobiGaussidelConvergen = []

    x = np.random.rand((dim))
    while len(jacobiGaussidelConvergen)< cantidad_matrices:
        size = dim
        
        # Obtener la descomposición en valores singulares de la matriz
        A = skd.make_spd_matrix(dim)

        # A = np.random.rand(dim,dim)
        #A = np.identity(dim)
        #x = np.random.rand(dim)
        #A = generar_sdp(dim)
        b = A @ x

        # f = open( "jacobi_matriz"  + ".txt", "w")
        with open('matriz.txt', 'w') as f:
            # Escribir las dimensiones en la primera línea
            f.write(f'{dim} {dim}\n')
            # Escribir la matriz en el resto del archivo
            np.savetxt(f, A)
        with open('solucion.txt', 'w') as f:
            # Escribir las dimensiones en la primera línea
            f.write(f'{dim}\n')
            # Escribir la matriz en el resto del archivo
            np.savetxt(f, b)
        with open('elX.txt', 'w') as f:
            # Escribir las dimensiones en la primera línea
            f.write(f'{dim}\n')
            # Escribir la matriz en el resto del archivo
            np.savetxt(f, x)
        os.system("./jacobi matriz.txt "  + "solucion.txt "+ "testeo " + "elX.txt") # O(m^4) siendo m la cantidad de pixeles de cada imagen
        converge = np.loadtxt("converge.txt")
        if converge[0]== 1 and converge[2] == 1 :
            jacobiGaussidelConvergen.append(A)
    np.save("jacobigausSeidelConvergen"+ str(dim), jacobiGaussidelConvergen)










def testeo_teraciones_error(dim, iteraciones ): # pide exista "jacobigausSeidelConvergen"+str(dim)
    ####
    """
    x = iteraciones, y= distancia a la respuesta correcta
    Jacobi y gaussidel
    
    """
    ####
    lista_convergen = np.load("jacobigausSeidelConvergen"+str(dim)+".npy")
    promedio_errores_por_itereracion_jacobi_matricial = np.zeros(iteraciones)
    promedio_errores_por_itereracion_jacobi_sumatoria = np.zeros(iteraciones)
    promedio_errores_por_itereracion_gaussidel_matricial = np.zeros(iteraciones)
    promedio_errores_por_itereracion_gaussidel_sumatoria = np.zeros(iteraciones)
    n = len(lista_convergen)
    x = np.random.rand(dim)
    for A in lista_convergen:
        b = A @ x
        with open('matriz.txt', 'w') as f:
            # Escribir las dimensiones en la primera línea
            f.write(f'{dim} {dim}\n')
            # Escribir la matriz en el resto del archivo
            np.savetxt(f, A)
        with open('solucion.txt', 'w') as f:
            # Escribir las dimensiones en la primera línea
            f.write(f'{dim}\n')
            # Escribir la matriz en el resto del archivo
            np.savetxt(f, b)
        with open('elX.txt', 'w') as f:
            # Escribir las dimensiones en la primera línea
            f.write(f'{dim}\n')
            # Escribir la matriz en el resto del archivo
            np.savetxt(f, x)
        os.system("./jacobi matriz.txt "  + "solucion.txt "+ "testeo " + "elX.txt")
        errores = np.loadtxt("error.txt")

        vector_error_completado = np.concatenate((errores[0], np.repeat( errores[0][len(errores[0])-1], iteraciones-len(errores[0]))))
        promedio_errores_por_itereracion_jacobi_matricial += vector_error_completado/n

        vector_error_completado = np.concatenate((errores[1], np.repeat( errores[1][len(errores[1])-1], iteraciones-len(errores[1]))))
        promedio_errores_por_itereracion_jacobi_sumatoria += vector_error_completado/n
        
        vector_error_completado = np.concatenate((errores[2], np.repeat( errores[2][len(errores[2])-1], iteraciones-len(errores[2]))))
        promedio_errores_por_itereracion_gaussidel_matricial += vector_error_completado/n
        
        vector_error_completado = np.concatenate((errores[3], np.repeat( errores[3][len(errores[3])-1], iteraciones-len(errores[3]))))
        promedio_errores_por_itereracion_gaussidel_sumatoria += vector_error_completado/n
    fig = plt.figure()
    plt.plot(errores[0], label="Error Jacobi Matriz")
    plt.plot(errores[1], label="Error Jacobi Sumatoria")
    plt.plot(errores[2], label="Error Gauss Seidel Matriz")
    plt.plot(errores[3], label="Error Gauss Seidel Sumatoria")
    plt.xlabel("iteraciones")
    plt.ylabel("error")
    plt.legend()
    plt.show()
    fig.savefig("iteracioneserror.png")
    

dim = 5
iter = 100
cantidad_matrices = 5

generar_convergen_jacobi_gauss(dim, cantidad_matrices)
testeo_teraciones_error(dim, 100)