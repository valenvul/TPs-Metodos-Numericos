import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plotError(error):
    # Plot the error
    error1 = error[0]
    error2 = error[1]
    error3 = error[2]
    error4 = error[3]
    fig = plt.figure()
    plt.plot(error1, label="Error Jacobi Matriz")
    plt.plot(error2, label="Error Jacobi Sumatoria")
    plt.plot(error3, label="Error Gauss Seidel Matriz")
    plt.plot(error4, label="Error Gauss Seidel Sumatoria")
    plt.legend()
    plt.show()
    # Save the plot
    fig.savefig("error.png")


error = np.loadtxt("error.txt")
print (error.shape)
iteraciones = np.loadtxt("iteraciones.txt")
tiempo = np.loadtxt("tiempo.txt")
print (tiempo.shape)
print (iteraciones.shape)
plotError(error)
