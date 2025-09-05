#include <eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <chrono>

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;



//Metodo de la potencia, recibe una matriz A, un entero con lal cantidad de interaciones
double powerIteration(const Eigen::MatrixXd& A, int iter, float tol, Eigen::VectorXd& x,  double lambda)
{
    //Obetener A.shape
    int size = (A).rows();
    Eigen::VectorXd res (size);
    x = Eigen::VectorXd::Random (size);
    bool cortoAntes = false;
    for (int i = 0; i < iter; i++)  // Complejidad O(iter)
    {
        res = (A)*(x);
        res.normalize();
        //Si el coseno de similar es parecido cortar
        float norm_diff = (res - (x)).norm();
        float norm_diff2 = (res + (x)).norm();
        if (norm_diff < tol || norm_diff2 < tol) {
            cortoAntes = true;
            std::cout << "Corto antes:"<< i << std::endl;
            break;
        }
        x = res;
    }
    if (!cortoAntes)
    {   
        x = Eigen::VectorXd::Random (size);
        std::cout << "No corto antes" << std::endl;
        powerIteration(A, iter, tol, x, lambda);     
    }
    
    lambda = (x).transpose() * (A) * (x);
    return lambda;

}

std::tuple<std::vector<Eigen::VectorXd>,Eigen::MatrixXd, std::vector<double> > eigen(const Eigen::MatrixXd& A, int iter, float tol,int k){
    std::vector <double> eigenvalues;
    std::vector <Eigen::VectorXd> eigenvectors;

    Eigen::VectorXd vector =  Eigen::VectorXd(); //Complejidad O(1)

    
    Eigen::MatrixXd A_copia =  Eigen::MatrixXd((A).rows(), (A).cols()); //Complejidad O(1)
    
    A_copia = A;
    
    for (int i = 0; i < k; i++) // Complehidad O(n)
    {
        double lambda = powerIteration(A_copia, iter, tol, vector,  lambda);

        A_copia -= (lambda) * (vector) * (vector).transpose();  //Complejidad O(n^2)
        eigenvalues.push_back(lambda); //Complejidad O(1)
        eigenvectors.push_back(vector); //Complejidad O(1)
        std::cout << "Iteracion: " << i << std::endl;
    }

    std::tuple<std::vector<Eigen::VectorXd>,Eigen::MatrixXd, std::vector<double> > result;
    result = std::make_tuple(eigenvectors, A_copia, eigenvalues);
    return result;
}
Eigen::MatrixXd leerMatrix(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin) {
        throw std::runtime_error("Error: no se pudo abrir el archivo " + filename);
    }

    int rows, cols;
    fin >> rows >> cols;

    Eigen::MatrixXd matrix(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fin >> matrix(i, j);
        }
    }

    fin.close();

    return matrix;
}


int matrizATexto(std::vector<Eigen::VectorXd> eigenvectors, std::vector<double> eigenvalues, std::string filename ){

    std::cout << "Escribiendo en archivo" << filename << std::endl;
    std::ofstream fout(filename  + "_autovectores.txt");
    if (!fout.is_open()) {
        std::cerr << "Error: no se puede abrir el archivo con los datos de salida" << std::endl;
        return 1;
    }
    int size = eigenvectors[0].size();
    // Escribimos la matriz
    for (int i = 0; i < size; i++) {
        for(auto j = 0; j < static_cast<int>(eigenvectors.size()); j++){
            fout << eigenvectors[j](i) << " ";
            if (j == static_cast<int>(eigenvectors.size()) - 1){
                fout << std::endl;
            }
        }
    }
    fout.close();

    std::ofstream fout2(filename + "_autovalores.txt");
    if (!fout2.is_open()) {
        std::cerr << "Error: no se puede abrir el archivo con los datos de salida" << std::endl;
        return 1;
    // Escribimos la matriz
    }
    for (auto i = 0; i < static_cast<int>(eigenvectors.size()); i++) {
        fout2 << eigenvalues[i] << " ";
    }
    fout2.close();
    return 0;
    }


int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << "Falta un argumento: " << argv[0] << " <filename> <iter>" << std::endl;
        return 1;
    }   

    std::string filename =  std::string(argv[1]); 
    auto start = std::chrono::high_resolution_clock::now();
    // Leer una matriz en texto y pasarla a MatrixXd, despues llamar a eigen y escribir sus eigenvalues en txt y en otro eigenvectores
    MatrixXd A = leerMatrix(filename + ".txt");
    std::cout << filename << std::endl;
    // Quiero que la tol se pase por argumento  en os
    int cant_autovectores = std::stoi(argv[2]);
    float tol = 1e-8;

    std::tuple<std::vector<Eigen::VectorXd>,Eigen::MatrixXd, std::vector<double> > x = eigen(A, 10000, tol,cant_autovectores);
    std::vector<Eigen::VectorXd> eigenvectors = std::get<0>(x);
    std::vector<double> eigenvalues = std::get<2>(x);
    //metodoPotencia(A, x, lambda, iter);
    matrizATexto(eigenvectors, eigenvalues, filename );


    // Fin del cronómetro
    auto end = std::chrono::high_resolution_clock::now();

    // Cálculo de la duración en segundos
    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();

    std::cout << "Tiempo tardado: " << seconds << " segundos" << std::endl;
    
    return 0;
}
