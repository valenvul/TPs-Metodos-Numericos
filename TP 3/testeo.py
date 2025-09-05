import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn.datasets as skd
""""
Archivo para comprobar que los métodos funcionen correctamente, se usan matrices conocidas 

"""

# Testear con distintos tipos de matrices
def matrizSDP(dim):
    A = np.random.randint(-10, 11, (dim, dim))
    b = np.abs(A)
    np.fill_diagonal(A, np.abs(b.sum(axis=1)) + np.random.randint(1, 10))
    A = A @ A.T
    return A

def matrizDiagonalDominante(dim):
    A = np.random.randint(-10, 11, (dim, dim))
    b = np.abs(A)
    np.fill_diagonal(A, np.abs(b.sum(axis=1)) + np.random.randint(15, 20))
    return A

def jacobiConverge(dim, n):
    """
    Funcion que busca que el radio espectral de la matriz de jacobi sea menor a 1
    
    """
    if n == 600:
        return  matrizDiagonalDominante(dim)
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
        return jacobiConverge(dim, n + 1)

def callMetodos(A, b,i ,x):
    
    with open('matriz.txt', 'w') as f:
        # Escribir las dimensiones en la primera línea
        f.write(f'{i} {i}\n')
        # Escribir la matriz en el resto del archivo
        np.savetxt(f, A)    
    with open('solucion.txt', 'w') as f:
        # Escribir las dimensiones en la primera línea
        f.write(f'{i}\n')
        # Escribir la matriz en el resto del archivo
        np.savetxt(f, b)
    with open('elX.txt', 'w') as f:
        # Escribir las dimensiones en la primera línea
        f.write(f'{i}\n')
        # Escribir la matriz en el resto del archivo
        np.savetxt(f, x)

    os.system("./jacobi matriz.txt "  + "solucion.txt "+ "testeo " + "elX.txt") # O(m^4) siendo m la cantidad de pixeles de cada imagen

def tiempoMatrices():
    exponencial = [3,10,50,100,250]
    tiempoSDPJacobi = np.zeros(len(exponencial), dtype=float)

    tiempoSDPGaussSeidel = np.zeros(len(exponencial))
    tiempoSDPLU = np.zeros(len(exponencial))

    tiempoDiagonalDominanteJacobi = np.zeros(len(exponencial))
    tiempoDiagonalDominanteGaussSeidel = np.zeros(len(exponencial))
    tiempoDiagonalDominanteLU = np.zeros(len(exponencial))

    tiempoJacobiConv = np.zeros(len(exponencial))
    tiempoJacobiGaussSeidel = np.zeros(len(exponencial))
    tiempoJacobiLU = np.zeros(len(exponencial))

    # Testear con distintos tipos de matrices, y ver el tiempo que tarda en resolverlas
    for i in range(len(exponencial)):

        for k in range(10):
            sdp = matrizSDP(exponencial[i])
            diagonalDom = matrizDiagonalDominante(exponencial[i])
            jacobiConv = jacobiConverge(exponencial[i], 0)
            x = np.random.randint(1, 11, (exponencial[i], 1))
            b = sdp @ x
            callMetodos(sdp, b, exponencial[i], x)
            tiempo = np.loadtxt("tiempo.txt") 
            tiempoSDPJacobi[i] += tiempo[0] / 10
            tiempoSDPGaussSeidel[i] += tiempo[3] / 10
            tiempoSDPLU[i] += tiempo[4] / 10
            b = diagonalDom @ x 
            callMetodos(diagonalDom, b, exponencial[i], x)
            tiempo = np.loadtxt("tiempo.txt")
            tiempoDiagonalDominanteJacobi[i] += tiempo[0] / 10
            tiempoDiagonalDominanteGaussSeidel[i] += tiempo[3] / 10
            tiempoDiagonalDominanteLU[i] += tiempo[4] / 10
            b = jacobiConv @ x
            callMetodos(jacobiConv, b, exponencial[i], x)
            tiempo = np.loadtxt("tiempo.txt")
            tiempoJacobiConv[i] += tiempo[0] / 10
            tiempoJacobiGaussSeidel[i] += tiempo[3] / 10
            tiempoJacobiLU[i] += tiempo[4] / 10
    # Plotear los tiempos
        print("Termino de testear con matrices de tamaño: ", exponencial[i])
    fig = plt.figure()
    plt.plot(exponencial, tiempoSDPJacobi, label="Tiempo SDP Jacobi")
    plt.plot(exponencial, tiempoSDPGaussSeidel, label="Tiempo SDP Gauss Seidel")
    plt.plot(exponencial, tiempoSDPLU, label="Tiempo SDP LU")
    plt.legend()
    plt.show()
    # Save the plot
    fig.savefig("tiempoSDP.png")
    fig = plt.figure()
    plt.plot(exponencial, tiempoDiagonalDominanteJacobi, label="Tiempo Diagonal Dominante Jacobi")
    plt.plot(exponencial, tiempoDiagonalDominanteGaussSeidel, label="Tiempo Diagonal Dominante Gauss Seidel")
    plt.plot(exponencial, tiempoJacobiLU, label="Tiempo Matriz LU para sistema jacobi converge")
    plt.legend()
    plt.show()
    # Save the plot
    fig.savefig("tiempoDiagonalDominante.png")
    fig = plt.figure()
    plt.plot(exponencial, tiempoJacobiConv, label="Tiempo Diagonal Dominante")
    plt.plot(exponencial, tiempoJacobiGaussSeidel, label="Tiempo SDP Gauss Seidel")
    plt.plot(exponencial, tiempoJacobiLU, label="Tiempo Diagonal Dominante Gauss Seidel")
    plt.legend()
    plt.show()
    # Save the plot
    fig.savefig("tiempoDiagonalDominante.png")

def callC(A,b,size):
    with open('matriz.txt', 'w') as f:
        # Escribir las dimensiones en la primera línea
        f.write(f'{size} {size}\n')
        # Escribir la matriz en el resto del archivo
        np.savetxt(f, A)    
    with open('solucion.txt', 'w') as f:
        # Escribir las dimensiones en la primera línea
        f.write(f'{size}\n')
        # Escribir la matriz en el resto del archivo
        np.savetxt(f, b)
    os.system("./jacobi matriz.txt "  + "solucion.txt") 
    error = np.loadtxt("error.txt")
    x = np.loadtxt("resultados.txt")
    converge = np.loadtxt("converge.txt")
    for i in range(len(x)):
        print(A @ x[i])
        if i < 4:
            if converge[i] == 1:
                print("Converge")
                print(np.allclose(A @ x[i], b))  
    # Limpiar los archivos
    
    os.system("rm matriz.txt")
    os.system("rm solucion.txt")
    os.system("rm error.txt")
    os.system("rm resultados.txt")

def sistemaSinEcuacion():
    A = np.array([[2, 1, 3],
                [1, -2, 4],
                [3, 5, 6]])

    b = np.array([5, 2, 1])

    x = np.linalg.solve(A, b)
    size = A.shape[0]
    callC(A,b,size)

def gausConverge():
    A = np.array([[1, 2, -2],
                [1, 1, 1],
                [2, 2, 1]])

    b = np.array([4, -1, 1])
    size = A.shape[0]
    callC(A,b,size)

def ambosConvergen():
    A = np.array([
        [5, 2, -2],
        [1, 3, 1],
        [2, 2, 6]])

    b = np.array([4, -1, 1])
    size = A.shape[0]
    callC(A,b,size)

def jacobiDiverge():
    A = np.array([
        [1, -0.5, 0.5],
        [1, 1, 1],
        [-0.5, -0.5, 1]])

    b = np.array([4, -1, 1])
    size = A.shape[0]
    callC(A,b,size)

def error(A,x ,b, converge  ):
    res = np.array([0,0,0,0,0])
    for i in range(len(x)):
            #use ecm
            if i < 4:
                if converge[i] == 1:
                    res[i] = np.linalg.norm(A @ x[i] - b)/ np.linalg.norm(b)            
            else:
                res[i] = np.linalg.norm(A @ x[i] - b)/ np.linalg.norm(b)
    return res


def erroresDisintoTipo():
    """
    Generar un grafico de lineas que compare los errores entre el resultado final y la solucion para distintos tipo de matrices
    y distintos tamano de matrices.
    """
    exponencial = [2,4,8,16,32,64,128,256,512]

    errorSDPJacobi = np.zeros(len(exponencial))
    errorSDPJacobiSum = np.zeros(len(exponencial))
    errorSDPGaussSeidel = np.zeros(len(exponencial))
    errorSDPGaussSeidelSum = np.zeros(len(exponencial))
    errorSDPLU = np.zeros(len(exponencial))

    errorDiagonalDominanteJacobi = np.zeros(len(exponencial))
    errorDiagonalDominanteJacobiSum = np.zeros(len(exponencial))
    errorDiagonalDominanteGaussSeidel = np.zeros(len(exponencial))
    errorDiagonalDominanteGaussSeidelSum = np.zeros(len(exponencial))
    erroroDiagonalDominanteLU = np.zeros(len(exponencial))

    errorJacobiConv = np.zeros(len(exponencial))
    errorJacobiConvSum = np.zeros(len(exponencial))
    errorJacobiGaussSeidel = np.zeros(len(exponencial))
    errorJacobiGaussSeidelSum = np.zeros(len(exponencial))
    errorJacobiLU = np.zeros(len(exponencial))
    iter = 10
    for i in range(len(exponencial)):
        for k in range(iter):
            sdp = skd.make_spd_matrix(exponencial[i])
            diagonalDom = matrizDiagonalDominante(exponencial[i])
            jacobiConv = jacobiConverge(exponencial[i], 0)
            x = np.random.randint(1, 11, (exponencial[i], 1))
            b = sdp @ x
            
            # Testear con Diagonal Dominante -> Gauss converge y Jacobi converge
            b = diagonalDom @ x 
            callMetodos(diagonalDom, b, exponencial[i], x)
            converge = np.loadtxt("converge.txt")
            x_res = np.loadtxt("resultados.txt")
            errores = error(diagonalDom, x_res, b, converge)
            
            errorDiagonalDominanteJacobi[i] += errores[0] / iter
            errorDiagonalDominanteJacobiSum[i] += errores[1] / iter
            errorDiagonalDominanteGaussSeidel[i] += errores[2] / iter
            errorDiagonalDominanteGaussSeidelSum[i] += errores[3] / iter
            erroroDiagonalDominanteLU[i] += errores[4] / 10
            

    
    plt.plot(exponencial, errorDiagonalDominanteJacobi, label="Diagonal Dominante Jacobi")
    plt.plot(exponencial, errorDiagonalDominanteJacobiSum, label="Diagonal Dominante Jacobi Sum")
    plt.plot(exponencial, errorDiagonalDominanteGaussSeidel, label="Diagonal Dominante Gauss Seidel")
    plt.plot(exponencial, errorDiagonalDominanteGaussSeidelSum, label="Diagonal Dominante Gauss Seidel Sum")
    plt.plot(exponencial, erroroDiagonalDominanteLU, label="Diagonal Dominante LU")
    # X = tamaño
    # Y = error
    plt.xlabel("Tamaño")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
    


#tiempoMatrices()
#SistemaSinEcuacion()
#   ("\nTesteando Sistema sin ecuacion")
print("\nTesteando Gauss converge \n")
#gausConverge()
print("\nTesteando Ambos convergen \n")
#ambosConvergen()
print("\nTesteando Jacobi diverge \n")
#jacobiDiverge()
print("\nTesteando errores distintos tipos \n")
erroresDisintoTipo()
    
