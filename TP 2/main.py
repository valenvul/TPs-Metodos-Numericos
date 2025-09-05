import pca
import numpy as np
import matplotlib.pyplot as plt
import similaridad
import dosdpca
import time


X, cant_filas, cant_columnas = pca.cargar_imagenes()
Xc =  X - np.mean(X, axis=0)

media, autovectores_pca, autovalores_pca, matriz_covarianza  = pca.calcular_pca(X,500, calcular_autovectores= False)


def comprimirPersona(X, k, autovectores_pca,l):
    # 
    #size promedios 
    promedios = []
    
    for k in range(1,16):
        ims = []
        D = []
        if l == 1:
            rango = 1
        else:
            rango = 9
            
        for j in range(0, rango):
            for i in range(0,9):    
                x_comprimida = pca.comprimir_imagen(X, autovectores_pca, k*25, (10*j) + i)
                ims.append(x_comprimida)    
        D = np.vstack(ims)
        R = np.corrcoef(D)
        R = similaridad.similaridad_PCA(D)
        promedios.append(np.mean(R))
    # dividir a promedios por 9
    

    return promedios

def graficarPromediosSimilaridad(X, k, autovectores_pca):

    promedios1 = comprimirPersona(Xc, 100, autovectores_pca,1)
    promedios2 =comprimirPersona(Xc, 100, autovectores_pca,2)

    plt.plot(range(25, 400, 25), promedios1, label='Promedio persona única')
    plt.plot(range(25, 400, 25), promedios2, label='Promedio personas múltiples')
    plt.xlabel('Cantidad de autovectores')
    plt.ylabel('Promedio de correlación')
    plt.legend()
    plt.show()


# CALCULAR DOSDPCA
imgs = dosdpca.cargar_imagenes_matrices()
imgs_mean = imgs - np.mean(imgs, axis = 0)
autovectores_dosdpca, autovalores_dosdpca = dosdpca.calcular_dosdpca(imgs,imgs[0].shape[1], calcular_autovectores= False)
"""
# Graficar similiritudes
graficarPromediosSimilaridad(Xc, 100, autovectores_pca)
similaridad.comparaciones(np.arange(5,80,5), autovectores_dosdpca,[10, 20, 50, 100] ,autovectores_pca, similaridad.separandoIdentidades() )

# Graficar eigenfaces
pca.graficar_eigenFaces(autovectores_pca, cant_filas, cant_columnas)
"""
dosdpca.graficar_eigenFaces(autovectores_dosdpca)

# Reconstruir imagen
#pca.graficar_imagenes_reconstruidas(Xc, autovectores_pca,cant_filas, cant_columnas)
dosdpca.reconstruir_imagenes_vk(imgs_mean, autovectores_dosdpca)


def tiempoReconstruccion(X, x_mean, Xc,img_pca, autovectores_pca, autovectores_dospca):

    erroresPca = []
    tiemposPca = []
    erroresDosPca = []
    tiemposDosPca = []
    for i in range(15,90, 5):
        x_comprimida = dosdpca.comprimir_imagen(x_mean[10], autovectores_dospca[:,0:i])
        start_time = time.time()
        x_rec = dosdpca.reconstruir_imagen(x_comprimida,  autovectores_dospca[:,0:i])
        end_time = time.time()

        elapsed_time = end_time - start_time
        tiemposDosPca.append(elapsed_time)
        erroresDosPca.append(errorCuadraticoMedio(x_mean[10], x_rec))

    for i in range(25,390,30):
        componentes_principales = autovectores_pca[:, :i]
        Z = np.dot(Xc, componentes_principales)
        start_time = time.time()
        x_rec = np.dot(Z, componentes_principales.T)
        img_unzipped    = np.real(x_rec[0])
        end_time = time.time()

        elapsed_time = end_time - start_time
        tiemposPca.append(elapsed_time)
        erroresPca.append(errorCuadraticoMedio(img_pca, img_unzipped))
    erroresDosPca = np.array(erroresDosPca)
    erroresPca = np.array(erroresPca)
    tiemposDosPca = np.array(tiemposDosPca)
    tiemposPca = np.array(tiemposPca)

    # Plot the arrays in ej x tiempos and y errores
    #plt.scatter(tiemposPca, erroresPca, label='PCA')
    plt.scatter(tiemposDosPca, erroresDosPca, label='DosPCA')
    plt.xlabel('Tiempos')
    plt.ylabel('Errores')
    plt.title('Errores vs Tiempos DOSPCA ')
    plt.legend()
    plt.show()
    plt.plot(range(15,90,5),tiemposDosPca)
    plt.title("Tiempo por iteracion 2DPCA")
    plt.show()
    plt.scatter(tiemposPca, erroresPca, label='PCA')
    plt.xlabel('Tiempos')
    plt.ylabel('Errores')
    plt.title('Errores vs Tiempos PCA')
    plt.show()
    plt.plot(range(25,390,30),tiemposPca)
    plt.title("Tiempo por iteracion PCA")
    plt.show()

def errorCuadraticoMedio(X, X_rec):
    """
    Calcula el error cuadrático medio entre la matriz original X y la matriz reconstruida X_rec.
    """
    error = np.mean(np.square(X - X_rec))
    return error

tiempoReconstruccion(imgs, imgs_mean, Xc,Xc[0], autovectores_pca, autovectores_dosdpca)


def erroresCuadraticos(X):
    """
    Calcula el error cuadrático medio entre la matriz original X y la matriz reconstruida X_rec.
    """
    errores_pca = []
    errores_2dpca = []

    for i in range(1, 400):
        x_comprimida = pca.comprimir_imagen(X, autovectores_pca,i ,0)
        x_rec = pca.reconstruir_imagen(x_comprimida, autovectores_pca, i)
        errores_pca.append(errorCuadraticoMedio(X[0], x_rec))
    errores = np.array(errores_pca)

    for i in range(1, 96):
        v_K = autovectores_dosdpca[:, 0:i]
        x_comprimida_2dpca = dosdpca.comprimir_imagen(imgs[0], v_K)
        x_rec_2dpca = dosdpca.reconstruir_imagen(x_comprimida_2dpca, autovectores_dosdpca)
        errores_2dpca.append(errorCuadraticoMedio(imgs[0], x_rec_2dpca))

    errores_2dpca = np.array(errores_2dpca)
    return  errores, errores_2dpca

def errorSinEntrenar():
    """
    Calcula el error cuadrático medio entre la matriz original X y la matriz reconstruida X_rec de una imagen que no fue parte
    del dataset.
    """
    path = 'ImagenesCaras/s41/1.pgm'
    img = plt.imread(path) 
    img_2dpca = img
    img = img.reshape(1, -1)
    autovectores_nocompletos = np.loadtxt("matriz_autovectores_reducidos.txt")

    error = []
    for i in range(25,390):
        autovectores_nocompletos_k = autovectores_nocompletos[:, 0:i]
        Z = np.dot(img, autovectores_nocompletos_k)
        x_rec_sin_entrenar = np.dot(Z, autovectores_nocompletos_k.T)
        error.append(errorCuadraticoMedio(img, x_rec_sin_entrenar))
    error = np.array(error)
    autovectores_nocompletos = np.loadtxt("matriz_autovectores_reducidos.txt")
    error_2dpca = []
    autovectores_nocompletos = np.loadtxt("matriz_2dpca_reducida_autovectores.txt")
    for j in range(5,96):
        x_comprimida_sin_entrenar = dosdpca.comprimir_imagen(img_2dpca, autovectores_nocompletos[:, 0:j])
        x_rec_sin_entrenar_2dpca = dosdpca.reconstruir_imagen(x_comprimida_sin_entrenar, autovectores_nocompletos[:, 0:j])
        error_2dpca.append(errorCuadraticoMedio(img_2dpca, x_rec_sin_entrenar_2dpca))
    error_2dpca = np.array(error_2dpca)
    return error, error_2dpca

def imagenSinEntrenar():
    """
    Reconstruye una imagen de la cual la cara no fue parte del dataset y la compara cuando si.
    """
    path = 'ImagenesCaras/s41/1.pgm'
    img = plt.imread(path) 
    img_2dpca = img
    img = img.reshape(1, -1)

    autovectores_nocompletos = np.loadtxt("matriz_autovectores_reducidos.txt")
    autovectores_nocompletos = autovectores_nocompletos[:, 0:389]
    Z = np.dot(img, autovectores_nocompletos)
    x_rec_sin_entrenar = np.dot(Z, autovectores_nocompletos.T)
    x_comprimida_entrenada = pca.comprimir_imagen(X, autovectores_pca,389 ,350)
    x_rec_entrenada = pca.reconstruir_imagen(x_comprimida_entrenada, autovectores_pca, 389)
    #Mismo usando 2dpca

    autovectores_nocompletos = np.loadtxt("matriz_2dpca_reducida_autovectores.txt")
    v_K = autovectores_dosdpca[:, 0:50]
    x_comprimida_sin_entrenar = dosdpca.comprimir_imagen(img_2dpca, autovectores_nocompletos[:, 0:50])
    x_rec_sin_entrenar_2dpca = dosdpca.reconstruir_imagen(x_comprimida_sin_entrenar, autovectores_nocompletos[:, 0:50])
    x_comprimida_entrenada_2dpca = dosdpca.comprimir_imagen(img_2dpca, v_K)
    x_rec_entrenada_2dpca = dosdpca.reconstruir_imagen(x_comprimida_entrenada_2dpca, autovectores_dosdpca[:, 0:50])
    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))

    # Plot the first image in the first subplot
    axs[0,0].imshow(x_rec_sin_entrenar.reshape(cant_filas,cant_columnas), cmap='gray')
    axs[0,0].set_title('Sin entrenar, error : ' + str(errorCuadraticoMedio(img, x_rec_sin_entrenar)))

    # Plot the second image in the second subplot
    axs[0,1].imshow(x_rec_entrenada.reshape(cant_filas,cant_columnas), cmap='gray')
    axs[0,1].set_title('Entrenada, error : ' + str(errorCuadraticoMedio(X[350], x_rec_entrenada)))

    axs[1,0].imshow(x_rec_sin_entrenar_2dpca, cmap='gray')
    axs[1,0].set_title('Sin entrenar, error : ' + str(errorCuadraticoMedio(img_2dpca, x_rec_sin_entrenar_2dpca)))

    # Plot the second image in the second subplot
    axs[1,1].imshow(x_rec_entrenada_2dpca, cmap='gray')
    axs[1,1].set_title('Entrenada, error : ' + str(errorCuadraticoMedio(img_2dpca, x_rec_entrenada_2dpca)))
    

    # Show the plot
    plt.show()

def plotErrorSinEntrenar():
    error, error_2dpca = errorSinEntrenar()
    plt.plot(error, label='PCA')
    plt.plot(error_2dpca, label='2DPCA')
    plt.xlabel('Cantidad de componentes principales')
    plt.ylabel('Error cuadrático medio')
    plt.legend()
    plt.show()

def correlaciones(pca_cant1, pca_cant2, pca_cant3, dosdpca_cant1, dosdpca_cant2, dosdpca_cant3  ):

    """
    
    """

    pca_Vk = autovectores_pca[:, :pca_cant3]
    dosdpca_Vk = autovectores_dosdpca[:, :dosdpca_cant3]
    dosdpca_zip_3 = [] 
    dosdpca_zip_2 = [] 
    dosdpca_zip_1 = [] 
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)    
    ######
    for i in range(len(imgs)):
        img_v = (imgs[i] @ dosdpca_Vk)
        dosdpca_zip_3.append( img_v.reshape(1, -1))
    dosdpca3 = np.vstack(dosdpca_zip_3)
    dosdpca3 = similaridad.similaridad_PCA(dosdpca3)

    axs[0, 2].pcolor(dosdpca3, cmap='GnBu')
    axs[0, 2].set_title('2DPCA :' + str(dosdpca_cant3))
    
    ######

    for i in range(len(imgs)):
        img_v = (imgs[i] @ dosdpca_Vk)
        dosdpca_zip_2.append( img_v.reshape(1, -1))
    dosdpca2 = np.vstack(dosdpca_zip_2)
    dosdpca2 = similaridad.similaridad_PCA(dosdpca2)
    
    axs[0, 1].pcolor(dosdpca2, cmap='GnBu')
    axs[0, 1].set_title('2DPCA :'+ str(dosdpca_cant2))

    
    ######
    for i in range(len(imgs)):
        img_v = (imgs[i] @ dosdpca_Vk)
        dosdpca_zip_1.append( img_v.reshape(1, -1))
    dosdpca1 = np.vstack(dosdpca_zip_1)
    dosdpca1 = similaridad.similaridad_PCA(dosdpca1)

    axs[0, 0].pcolor(dosdpca1, cmap='GnBu')
    axs[0, 0].set_title('2DPCA :'+ str(dosdpca_cant1))

    ######
    pca3 = X @ pca_Vk
    pca3 = similaridad.similaridad_PCA(pca3)
    axs[1, 2].pcolor(pca3, cmap='GnBu')
    axs[1, 2].set_title('PCA : '+ str(pca_cant3))
    pca_Vk = pca_Vk[:, 0 : pca_cant2]
    #####
    pca2 = X @ pca_Vk
    pca2 = similaridad.similaridad_PCA(pca2)
    axs[1, 1].pcolor(pca2, cmap='GnBu')
    axs[1, 1].set_title('PCA : '+str(pca_cant2))
    pca_Vk = pca_Vk[:, 0 : pca_cant1]
    #####
    pca1 = X @ pca_Vk
    pca1 = similaridad.similaridad_PCA(pca1)
    axs[1, 0].pcolor(pca1, cmap='GnBu')
    axs[1, 0].set_title('PCA : '+str(pca_cant1))
    #####

    fig.colorbar(axs[0, 2].pcolor(dosdpca3, cmap='GnBu'), ax=axs.ravel().tolist())

    plt.show()


    return

def plotear_arreglo_autovalores(vec):
    for i in range(len(vec)):
        vec[i] = np.absolute(vec[i])
    plt.plot(vec)
    plt.xlabel('autovalores')
    plt.ylabel('magnitud')
    plt.show()


######
# Experimentación y gráficos 
#  
plotErrorSinEntrenar()

#correlaciones(10, 100, 400, 3, 15, 55)

# XC = similaridad.similaridad_PCA(X)
# plt.imshow(XC, cmap='GnBu', origin='lower')
# plt.colorbar()
# plt.show()
# 
# plotear_arreglo_autovalores(autovalores_dosdpca)
# plotear_arreglo_autovalores(autovalores_pca)

