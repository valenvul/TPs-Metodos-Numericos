import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import similaridad



import os
from os import listdir




imgs = []

def cargar_imagenes_matrices():
    # Cargar las imágenes de la carpeta ImagenesCaras a una lista de imágenes como matrices
   
    paths = []
    imgs = []
    for path in sorted(list(Path('ImagenesCaras').rglob('*/*.pgm'))):  # O(n) siendo n la cantidad de imágenes
        paths.append(path)
        img = plt.imread(path) 
        imgs.append(img) 

    return imgs

   

def calcular_dosdpca(lista_imgs, k ,calcular_autovectores): # la lista tiene que no ser vacía
   """
   dada una lista de imágenes calcula la image scatter matrix y sus k autovectores con autovalores de mayor magnitud
   Devuelve:
   V = [X_1 | ... | X_k] siendo todos los X los autovectores de G, unitarios y ordenados de mayor a menor
   """
   if not calcular_autovectores :
      autovectores = np.loadtxt("matriz_2dpca_autovectores.txt")
      autovalores = np.loadtxt("matriz_2dpca_autovalores.txt")

      return autovectores, autovalores

   M = len(lista_imgs)  #cant imagenes
   a = len(lista_imgs[0]) #alto de cada imagen
   b = len(lista_imgs[0][0]) #ancho de cada imagen
   
   #G va a ser la matriz de covarianza de las imagenes
   G = np.zeros([b,b]) 
   img_promedio = np.zeros([a,b])
   
   #calculo la imagen promedio
   for img in lista_imgs:
      img_promedio += img                                        
   img_promedio = img_promedio / M

   #calculo la matriz de covarianza
   for img in lista_imgs:
      G += ( ((img - img_promedio).T )@( (img - img_promedio) ) )     
   G = G / M

   # Calcular los autovalores y autovectores de la matriz de covarianza
   # Hay que llamar a c++  para que calcule los autovalores y autovectores.
   # escribir la matriz en un archivo y llamar a c++ para que calcule los autovalores y autovectores
   filas, columnas = G.shape
   with open('matriz_2dpca.txt', 'w') as f:
      # Escribir las dimensiones en la primera línea
      f.write(f'{filas} {columnas}\n')
      # Escribir la matriz en el resto del archivo
      np.savetxt(f, G)
   #autovalores, autovectores = np.linalg.eig(matriz_covarianza)

   os.system("./metodoPotencia matriz_2dpca " + str(k))
   autovalores = np.loadtxt("matriz_2dpca_autovalores.txt")
   autovectores = np.loadtxt("matriz_2dpca_autovectores.txt")

   """
   Esto de acá es para hacer todos los gráficos
   for i in range(len(autovalores)):
      autovalores[i] = np.absolute(autovalores[i])
   plt.plot(autovalores)
   plt.xlabel('autovalores')
   plt.ylabel('magnitud')
   plt.show()
   
   plt.pcolor(np.corrcoef(G))
   plt.show()
   """
   #autovectores = autovectores.T


   return (autovectores, autovalores)
    



def comprimir_imagen(img, Vk): #comprime
   """
   Dada una matriz de autovectores V = [ X_1 | ... | X_n ] y una imagen A devuelve un P = AV 
   Args:
   autovectores_twodpca: V
   img : imagen a comprimir
   d: dimension a la que se busca proyectar

   Returns:
   P: vectores caracteristicos = [A @ X_1 | ... | A @ X_k]
   
   """
   P = img @ Vk
   """
   P = np.zeros((img.shape[0], Vk.shape[1]))
   print("Pshape :", P.shape)
   for i in range(k):
      Xi = Vk[:, i]
      P[:, i] = img.dot(Xi)
   """
   return P
   

def reconstruir_imagen(P, V):#descomprime
   
   """
   V  = [ X_1 | ... | X_k ] siendo X_1 el autovector con autovalor de mayor magnitud y el d-ésimo el de menor.      
   P  = [A @ X_1 | ... | A @ X_k] ( A siendo la imagen original)

   Returns:
   Aprox: Imagen aproximada 
   """
   if P.shape[1] == V.shape[1] :
      return P @ V.T
   Vk = V[: , :P.shape[1] ]
   return P @ Vk.T
   # Aprox = np.zeros((P.shape[0], Vk.shape[0]))

   # for i in range(Vk.shape[1]):
   #    Xk = Vk[:, i]
   #    Yk = P[:, i]
   #    Aprox += np.outer(Yk, Xk)

def reconstruir_imagenes_vk(img, V):
   """
   dado una imagen, la comprime y reconstruye con distintos auutovalores
   """
   contador  = 1
   fig = plt.figure(figsize=(200, 200))
   # Iterar sobre diferentes valores de k y reconstruir las imágenes
   for j in range(5):
   
      for i in range(1, 16,3):
         if i == 13:
            i = 50
         componentes_principales = V[:, :i]
         # Proyectar el dataset sobre los autovectores principales para obtener los coeficientes
         Z = comprimir_imagen(img[j*15], componentes_principales)
         # Reconstruir el dataset usando los autovectores principales y los coeficientes
         X_rec = reconstruir_imagen(Z, componentes_principales)
      
         fig.add_subplot(5,5,contador)
         plt.imshow(X_rec, cmap='gray')
         plt.title(str(i+1))
         contador += 1
         plt.axis('off')
   plt.show()

def graficar_eigenFaces(V):
   fig = plt.figure(figsize=(12,8))
   for i in range(9):

      fig.add_subplot(3,3, i+1 )
      plt.imshow((V[:, i]).reshape(23,4), cmap='gray')
      plt.title("2dpca_eigenface : " + str(i))
      plt.axis('off')

   plt.show()
