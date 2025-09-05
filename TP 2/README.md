Para testear el metodo de la potencia se corre testMetodoPotencia que tiene distintas matrices, tambien se puede correr desde el archivo de c++. Como argumento recibe un archivo de texto que es la matriz a calcular y cuantos autovectores calcula

Antes de correr descomprimir los zip, la unión entre c++ y python se hace por os.
Al correr main.py se corren PCA y 2DPCA y los graficos, si no se quieren volver a calcular todos los autovalores de pca/2dpca , se puede pasar como parametro false en calcular_pca. En main hay funciones que calculan el error cuadratico medio entre una imagen original y su reconstruida. Luego erroresCuadraticos grafica el promedio de error de reconstrucción de distintos rostros para distinta cantidad de autovalores.

imagenSinEntrenar grafica una persona que fue dejada de lado antes de correr los test y su reconstrucción con PCA y 2DPCA

La cantidad de autovectores dados en los .txt son 410
No es recomendable correr el calculo de autovectores para PCA debido a su alto tiempo de ejecución, 2DPCA se puede correr sin problema.


Para compilar el codigo correr make, esto generara el ejecutable de metodoPotencia y también hace el unzip, luego se puede correr directamente los .py descomentando las funciones deseadas.
Para borrar el ejecutable y los .txt correr make clean