import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import similaridad
import os

def cargar_imagenes():
    # Cargar las imágenes de la carpeta ImagenesCaras

    paths = []
    imgs = []
    for path in sorted(list(Path('ImagenesCaras').rglob('*/*.pgm'))):  # O(n) siendo n la cantidad de imágenes
        paths.append(path)
        img = plt.imread(path) 
        cant_filas, cant_columnas = img.shape
        img_vector = img.reshape(1, -1)
        imgs.append(img_vector) 
    print("La shape de cada imagen es de: ", img_vector.shape)
    print("La cantidad de imagenes es de: ", len(imgs))

    # Apilar las imágenes en una matriz
    X = np.vstack(imgs)
    
    return X, cant_filas, cant_columnas

def calcular_pca(X, k, calcular_autovectores):
    """
    Calcula la matriz de covarianza y los autovectores de la misma
    Args:
        X: matriz de imágenes
        k: cantidad de autovectores a calcular
        calcular_autovectores: booleano que indica si se llama a la funcion de c++ para calcular los autovectores
    Returns:
        media: media del dataset
        X_centralizado: dataset centrado
        autovectoresprop: autovectores de la matriz de covarianza

    """


    # Calcular la media del dataset
    media = np.mean(X, axis=0) # O(n*m) siendo n la cantidad de imágenes y m la cantidad de pixeles de cada imagen

    # Centrar el dataset restando la media
    X_centralizado = X - media # O(n*m) siendo n la cantidad de imágenes y m la cantidad de pixeles de cada imagen

    # Calcular la matriz de covarianza del dataset centrado
    matriz_covarianza = similaridad.matriz_cov(X) # O(n*m^2) siendo n la cantidad de imágenes y m la cantidad de pixeles de cada imagen

    # Calcular los autovalores y autovectores de la matriz de covarianza
    # Hay que llamar a c++  para que calcule los autovalores y autovectores.
    # escribir la matriz en un archivo y llamar a c++ para que calcule los autovalores y autovectores
    
    
    print("Calculando autovectores y autovalores...")
    if calcular_autovectores:
        filas, columnas = matriz_covarianza.shape
        with open('matriz.txt', 'w') as f:
            # Escribir las dimensiones en la primera línea
            f.write(f'{filas} {columnas}\n')
            # Escribir la matriz en el resto del archivo
            np.savetxt(f, matriz_covarianza)
        os.system("./metodoPotencia matriz "  + str(k)) # O(m^4) siendo m la cantidad de pixeles de cada imagen
    
    
    autovectoresprop = np.loadtxt("matriz_autovectores.txt") # 
    autovalores = np.loadtxt("matriz_autovalores.txt")



    return media, autovectoresprop, autovalores, matriz_covarianza


def comprimir_imagen(X, autovectores, k , i):
    """
    Comprime una imagen usando los autovectores principales
    args:
        X: matriz de imágenes
        autovectores: autovectores de la matriz de covarianza
        k: cantidad de autovectores a usar para reconstruir las imágenes
        i: imagen a reconstruir
    ret:
        X_com : imagen comprimida
    """
    componentes_principales = autovectores[:, :k]
    # Proyectar el dataset sobre los autovectores principales para obtener los coeficientes
    Z = np.dot(X, componentes_principales)
    z_i = Z[i]
    return z_i

def reconstruir_imagen(z_i, autovectores, k):
    """
    Reconstruye una imagen usando los autovectores principales
    args:
        zi: imagen comprimida
        autovectores: autovectores de la matriz de covarianza
        k: cantidad de autovectores a usar para reconstruir las imágenes
    ret:
        X_rec : imagen reconstruida
    """
    componentes_principales = autovectores[:, :k]
    X_rec = np.dot(z_i, componentes_principales.T)
    return X_rec


def graficar_eigenFaces(autovectores, cant_filas, cant_columnas):
    """
    Grafica los 16 autovectores principales
    args:
        autovectores: autovectores de la matriz de covarianza
    """
    autovectores_principalesT = autovectores[:,  0:25]
    autovectores_principalesT = np.real(autovectores_principalesT).T
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))

    #fig = plt.figure(figsize=(10,10))
    for i in range(16):
        #fig.add_subplot(4,4,i+1)
        axs[i//4, i%4].imshow(autovectores_principalesT[i].reshape(cant_filas,cant_columnas), cmap='gray')
        axs[i//4, i%4].set_title("Imagen eigface " + str(i+1))
        axs[i//4, i%4].axis('off')
        #plt.imshow(autovectores_principalesT[i].reshape(cant_filas,cant_columnas), cmap='gray')
        #plt.axis('off')
        #plt.title("Imagen eigface " + str(i+1))
    plt.show()


def graficar_imagenes_reconstruidas(X,autovectores, cant_filas, cant_columnas):
    """
    Grafica las imágenes reconstruidas usando los autovectores principales \n
    args:
        X: matriz de imágenes
        media: media del dataset
        autovectores: autovectores de la matriz de covarianza
        k: cantidad de autovectores a usar para reconstruir las imágenes
        rango: cantidad de autovectores a saltar
        i: imagen a reconstruir
    """
    contador  = 1
    fig = plt.figure(figsize=(200, 200))
    # Iterar sobre diferentes valores de k y reconstruir las imágenes
    for j  in range(5):
        for i in range(1,400,80 ):
            componentes_principales = autovectores[:, :i]
            Z = np.dot(X, componentes_principales)
            X_rec = np.dot(Z, componentes_principales.T)
            img_rec = X_rec[j*15].reshape(cant_filas, cant_columnas)
            img_unzipped    = np.real(img_rec)
       
            fig.add_subplot(5,5,contador)
            plt.imshow(img_unzipped, cmap='gray')
            plt.title(str(i+1))
            contador += 1
            plt.axis('off')
    plt.show()

    return X_rec[j]

