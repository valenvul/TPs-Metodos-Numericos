
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import dosdpca

def matriz_cov(X):
  """
  partiendo de X un conjunto de datos dispuesto en columnas :
    X = [ x1 | ... | xn ] 
    devuelve matriz C tal que
    matriz_cov(X) = C
    con C //
    C_ij = cov(x_i , x_n)

  """
  m , n = X.shape  

  #primero tengo que centrar los datos
  #calculo la media para cada columna de X, es decir cada atributo (cada pixel)
  medias = []
  for j in range(n):
    u_j = np.average(X[:,j])
    medias.append(u_j)
  medias = np.array(medias)

  # a cada vector imagen (fila de X) le resto el vector de las medias
  Xc = []
  for i in range(m):
      X_i = X[i] - medias
      Xc.append(X_i)

  #calculo la matriz de covarianza
  Xc = np.array(Xc)
  C = (Xc.T @ Xc) / (n - 1)

  return C


# recibe X con m imagenes de n pixeles total, cada fila es una img
def similaridad_PCA(X): 
  #devuelve matriz R de similaridad 

  n = len(X[0])
  m = len(X)

  medias = []
  moduloX = abs(X)
  for j in range(n):
    u_j = np.average(moduloX[:,j])
    medias.append(u_j)
  medias = np.array(medias)

  Xc = []
  for i in range(m):
    X_i = X[i] - medias
    Xc.append(X_i)

  Xc = np.array(Xc)
  C = (Xc @ Xc.T) / (n-1)


  R = np.empty([m,m])
  i = 0
  while i < m:
    j=0
    while j < m:
        denominador = np.sqrt(C[i][i]*C[j][j])
        R[i][j] = C[i][j] / denominador
        j += 1
    i += 1
  return R


  # C = matriz_cov(X.T) 
  # # C_ij = cov( img_i , img_j )
  
  # n , m = (X.T).shape
  # R = np.empty([m,m])
  # print("shape R : ", R.shape)
  # i = 0
  # while i < m:
  #   j=0
  #   while j < m:
  #     denominador = np.sqrt(C[i][i]*C[j][j])
  #     R[i][j] = C[i][j] / denominador
  #     j += 1
  #   i += 1

  # return R

def prom_simil(R):
  prom_filas = []
  for i in range(len(R)):
      promedio = np.average(R[i])
      prom_filas.append(promedio)

  res = np.average(prom_filas)

  return res

def traerImagenes():
  paths = []
  imgs = []
  for path in sorted(list(Path('caras').rglob('*/*.pgm'))):
      paths.append(path)
      img = plt.imread(path)[::2,::2]/255
      img_vector = img.reshape(1, -1)  # Convierte la matriz en un vector de fila
      imgs.append(img_vector) 
  X = np.vstack(imgs)

  R = similaridad_PCA(X)
  plt.pcolor(R)

def identidades_paths():
    # Define the main folder
    main_folder = 'ImagenesCaras'

    # Create a list to store paths for each identity  
    id_paths = []
    # Iterate over the subfolders in the main folder
    for subfolder in sorted(Path(main_folder).iterdir()):
        if subfolder.is_dir():
            # Create a list to store paths for the current identity
            current_id_paths = []
            
            # Iterate over the image files in the subfolder
            for file_path in sorted(subfolder.glob('*.pgm')):
                current_id_paths.append(str(file_path))
            
            # Add the current identity's paths to the main list
            id_paths.append(current_id_paths)
    return id_paths


def separandoIdentidades():
    identidades = []
    id_paths = identidades_paths()
    for i in range(len(id_paths)):
        persona = []
        for k in range(len(id_paths[i])):
            img = plt.imread(id_paths[i][k]) / 255
            persona.append(img)
        identidades.append(persona)
    return identidades


def comparaciones(dosdpca_cantidades, dosdpca_V, pca_cantidades, pca_V, identidades ):
  ## se empieza con 2dpca y las que son todas iguales
  dosdpca_Vk = dosdpca_V 
  data_2dpca_iguales = []
  n = len(identidades)-1
  for k in reversed(dosdpca_cantidades):
    dosdpca_Vk = dosdpca_V[:, :k]
    prom = 0
    for persona in identidades:
       X = []
       for img in persona:
          X.append((dosdpca.comprimir_imagen(img, dosdpca_Vk)).reshape(1, -1))
       X = np.vstack(X)
      #  X = similaridad_PCA(X)
       X = similaridad_PCA(X)
       X = np.array(X)
       for xi in range(len(X.shape)):
            X[xi] = abs(X[xi])
       prom += np.mean(X) / n
    data_2dpca_iguales.append(prom)

  plt.plot(dosdpca_cantidades, np.flip(data_2dpca_iguales)) 
  ####
  dosdpca_Vk = dosdpca_V 
  data_2dpca_diferentes = []
  n = len(identidades)
  for k in reversed(dosdpca_cantidades):
    dosdpca_Vk = dosdpca_V[:, :k]
    prom = 0

    for l in range(10):
       X = []
       for i in range(len(identidades)):
          X.append((dosdpca.comprimir_imagen(identidades[i][l], dosdpca_Vk)).reshape(1, -1) )
       X = np.vstack(X)
       X = similaridad_PCA(X)
       X = np.array(X)
       for xi in range(len(X.shape)):
            X[xi] = abs(X[xi])
       prom += np.mean(X) #aca había un /n

    data_2dpca_diferentes.append(prom)

  for i in range(len(data_2dpca_diferentes)):
    data_2dpca_diferentes[i] = data_2dpca_diferentes[i] / n

  dosdpca_cantidades = np.array(dosdpca_cantidades)

  data_2dpca_diferentes = np.array( np.flip(data_2dpca_diferentes)  )
  plt.plot(dosdpca_cantidades, data_2dpca_diferentes) 
  plt.xlabel('cantidad autovectores')
  plt.ylabel('promedio del promedio de matrices de similaridad ')
  plt.legend(['iguales', 'diferentes'], loc='upper right')

  ####
  # Display the plot
  plt.show()