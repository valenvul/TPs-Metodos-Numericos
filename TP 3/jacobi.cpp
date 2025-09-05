#include <eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

struct Tupla{
    Eigen::VectorXd x;
    Eigen::VectorXd error;
    int iter;
    double tiempo;
    bool converge;
};

Eigen::MatrixXd tInf(Eigen::MatrixXd A){
    Eigen::MatrixXd tinf = A;
    tinf.setZero(A.rows(), A.cols());

    for (int i = 0; i < A.rows(); i++){
        for (int j = 0; j < A.cols(); j++){
            if (i > j){
                tinf(i,j) = A(i,j);
            }
        }
    }

    return tinf;
}

Eigen::MatrixXd tSup (Eigen::MatrixXd A){
    Eigen::MatrixXd tsup = A;
    tsup.setZero(A.rows(), A.cols());
    for (int i = 0; i < A.rows(); i++){
        for (int j = 0; j < A.cols(); j++){
            if (i < j){

                tsup(i,j) = A(i,j);
            }
        }
    }


    return tsup;
}

Tupla jacobi(Eigen::MatrixXd A, Eigen::VectorXd b, Eigen::VectorXd sol, int iter= 10, double tol = 0.000001,
     bool testeo = false ) {
    Tupla res;
    auto start = std::chrono::high_resolution_clock::now();
    // Preparo las variables, D, L, U
    Eigen::MatrixXd D = A.diagonal().asDiagonal();
    Eigen::MatrixXd L = A.triangularView<Eigen::Lower>();
    L = L - D;
    Eigen::MatrixXd U = A.triangularView<Eigen::Upper>();
    U = U - D;
    Eigen::VectorXd xk = Eigen::VectorXd::Random (A.rows());
    xk.setZero(A.rows());
    Eigen::VectorXd x = xk;
    Eigen::MatrixXd UL = -U - L; 
    Eigen::MatrixXd T = D.inverse() *(UL);
    Eigen::VectorXd c = D.inverse() * b; 
    Eigen::VectorXd error = Eigen::VectorXd::Random (iter);
    bool converge = false;
    error.setZero(iter);
    int i = 0;
    double dist;

    // Commienza a iterar
    for (i ; i < iter ; i++){        
        xk = T * x + c;
        dist =  (xk-x).norm();        // distancia de la última iteración con la recién calculada

        if (testeo){
            error[i] = (xk - sol).norm() ;
        }else{
            error[i] = dist;
        }
        
        //chequea si llegó*/
        if (i > 1 && dist < tol ){
            std::cout << "Converge en la iteracion jacobi: " << i << std::endl;
            converge = true;
            break;
        }
        
        x = xk;
    }
    if( !converge ){
        if ((A * xk - b ).norm()< tol){
            converge = true;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd errorCompleto = Eigen::VectorXd::Random (i);

    for (int j = 0; j < i; j++){
        errorCompleto[j] = error[j];
    }
    
    // Cálculo de la duración en segundos
    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();
    res.error = error;
    res.x = xk;
    res.iter = i;
    res.converge = converge;
    res.tiempo = seconds;
    return res;
}


Tupla sumatoriaJacobi(Eigen::MatrixXd A, Eigen::VectorXd b, Eigen::VectorXd sol, int iter= 10, double tol = 0.000001,
     bool testeo = false){
    Tupla res;
    auto start = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd xk = Eigen::VectorXd::Random (A.rows());
    xk.setZero(A.rows());

    Eigen::VectorXd xkant = Eigen::VectorXd::Random (A.rows());

    xkant.setZero(A.rows());
    Eigen::VectorXd error = Eigen::VectorXd::Random (iter);
    error.setZero(iter);
    bool converge = false;
    int i = 0;
    double dist;
    //hacer sumatoria sub_i
    Eigen::VectorXd x_ant2 = xk;
    for ( i; i < iter ; i++){
        for (int j = 0; j < A.cols(); j ++) {
            double sumatoria = 0;
            for (int k = 0; k < A.rows(); k++){
                if (k != j){
                    sumatoria += A(j,k) * xkant[k];
                }
            }
            xk[j] = 1/A(j,j) * (b(j)- sumatoria);
        }
        dist = (xk-xkant).norm();
        if (testeo){
            error[i] = (xk - sol).norm() ;
        }else{
            error[i] = dist;
        }

        if (dist < tol ){
            std::cout << "Converge en la iteracion sumatoria: " << i << std::endl;
            converge = true;
            break;
        }
        xkant = xk;
    }
    if( !converge ){
        if ((A * xk - b ).norm()< tol){
            converge = true;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd errorCompleto = Eigen::VectorXd::Random (i);

    for (int j = 0; j < i; j++){
        errorCompleto[j] = error[j];
    }
    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();
    res.error = error;
    res.x = xk;
    res.iter = i;
    res.tiempo = seconds;
    res.converge = converge;
    return res;
}

Tupla gauus(Eigen::MatrixXd A, Eigen::VectorXd b, Eigen::VectorXd sol, int iter= 10, double tol = 0.000001,
     bool testeo = false){
    Tupla res;
    auto start = std::chrono::high_resolution_clock::now();
    bool converge = false;
    Eigen::VectorXd error = Eigen::VectorXd::Random (iter);
    error.setZero(iter);    
    Eigen::MatrixXd D = A;
    D.setZero(A.rows(), A.cols());
    for (int i = 0 ; i < D.rows(); i++){
        D(i,i) = A(i,i);
    }
    Eigen::MatrixXd U = -tSup(A);
    Eigen::MatrixXd L = -tInf(A);

    Eigen::VectorXd xk = Eigen::VectorXd::Random (A.rows());
    xk.setZero(A.rows());
    Eigen::VectorXd x = xk;
    double dist;
    Eigen::MatrixXd tinf = D-L;

    Eigen::MatrixXd T = tinf.inverse() * (U) ;
    Eigen::VectorXd c = tinf.inverse() * b; 

    int i = 0;
    for  (i;i < iter ; i++){
            // xk = tinf.inverse() * (b + (U)*xk);
            xk = T*xk + c;
            dist = (x-xk).norm();

            if (testeo){
                error[i] = (xk - sol).norm() ;
            }else{
                error[i] = dist;
            }
            dist = (x-xk).norm();

            if (i > 1 && dist < tol ){
                std::cout << "Converge en la iteracion gauss: " << i << std::endl;
                converge = true;

                break;
            }
            x = xk;
    }
    if( !converge ){
        if ((A * xk - b ).norm()< tol){
            converge = true;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    


    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();
    res.x = xk;
    res.iter = i;
    res.tiempo = seconds;
    res.converge = converge;
    res.error = error;

    return res;
        //TP TERMINADO
}

Tupla sumgauss(Eigen::MatrixXd A, Eigen::VectorXd b, Eigen::VectorXd sol, int iter= 10, double tol = 0.000001,
     bool testeo = false){
    Tupla res;
    auto start = std::chrono::high_resolution_clock::now();
    bool converge = false;
    Eigen::VectorXd error = Eigen::VectorXd::Random (iter);
    error.setZero(iter);
    
    Eigen::VectorXd xk = Eigen::VectorXd::Random (A.rows());
    xk.setZero(A.rows());
    Eigen::VectorXd xkant = xk;
    double dist;

    
    //hacer sumatoria sub_i
    int i = 0;
    for (i; i < iter ; i++){

        for (int j = 0; j < A.rows(); j ++) {
            double sumatoria = 0;
            for (int k = 0; k < A.cols  ();k++){
                if (k != j){
                    sumatoria += A(j,k) * xk(k); // Los elementos modificados de xk van de 1 hasta i-1, pertenecen
                                                // al paso i+1
                }
            }
            xk[j] = 1/A(j,j) * (b[j]- sumatoria);
        }
        dist = (xk-xkant).norm();
        
        if (testeo){
            error[i] = (xk - sol).norm() ;
        }else{
            error[i] = dist;
        }
        if (dist < tol ){
            std::cout << "Converge en la iteracion sumatoria: " << i << std::endl;
            converge = true;

            break;
        }
        xkant = xk;
    }
    if( !converge ){
        if ((A * xk - b ).norm()< tol){
            converge = true;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();
    res.error = error;
    res.x = xk;
    res.iter = i;
    res.tiempo = seconds;
    res.converge = converge;
    
    return res;
}

Eigen::MatrixXd leerMatrix(std::string filename){
    //filename  + ".txt"

    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error: could not open input file " << filename << std::endl;
        return Eigen::MatrixXd::Zero(0, 0);
    }

    // Leer el número de filas y columnas
    int nrows, ncols;
    fin >> nrows >> ncols;


    Eigen::MatrixXd A(nrows, ncols);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            fin >> A(i, j);
        }
    }

    fin.close();
    return A;
}

Eigen::VectorXd leerSolucion(std::string filename){
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error: could not open input file " << filename << std::endl;
        return Eigen::MatrixXd::Zero(0, 0);
    }
    int nrows;
    fin >> nrows;
    Eigen::VectorXd b(nrows);
    for (int i = 0; i < nrows; i++) {
        fin >> b(i);
    }
    fin.close();
    return b;
}

int matrizATexto(Eigen::MatrixXd res, std::string filename ){

    std::ofstream fout(filename  + ".txt");
    if (!fout.is_open()) {
        std::cerr << "Error: no se puede abrir el archivo con los datos de salida" << std::endl;
        return 1;
    }
    int size = res.rows();
    // Escribimos la matriz
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < res.cols(); j++){
            fout << res(i,j) << " ";
        }
        fout << std::endl;
    }
    fout.close();
    return 0;
    }

int vectorATexto(Eigen::VectorXd vector, std::string filename){
    std::ofstream fout(filename  + ".txt");
    if (!fout.is_open()) {
        std::cerr << "Error: no se puede abrir el archivo con los datos de salida" << std::endl;
        return 1;
    }
    int size = vector.size();
    // Escribimos el vector
    for (int i = 0; i < size; i++) {
            fout << vector(i) << " ";
    }
    fout.close();
    return 0;
}



void guardarResultados(Tupla jaconi, Tupla sumJacobi, Tupla gaussi, Tupla sumatoriaGaussi, Eigen::VectorXd x, double seconds, bool testeo){
    Eigen::MatrixXd resultados(5, x.size());
    resultados.row(0) = jaconi.x;
    resultados.row(1) = sumJacobi.x;
    resultados.row(2) = gaussi.x;
    resultados.row(3) = sumatoriaGaussi.x;
    resultados.row(4) = x;
    matrizATexto(resultados, "resultados");
    
        int errorMax = jaconi.iter;
        for (int i = 0; i < 4; i++){
            if (errorMax < sumJacobi.iter){
                errorMax = sumJacobi.iter;
            }
            if (errorMax < gaussi.iter){
                errorMax = gaussi.iter;
            }
            if (errorMax < sumatoriaGaussi.iter){
                errorMax = sumatoriaGaussi.iter;
            }
        }
        Eigen::MatrixXd error(4, errorMax);
        error.row(0) = jaconi.error.head(errorMax);
        error.row(1) = sumJacobi.error.head(errorMax);
        error.row(2) = gaussi.error.head(errorMax);
        error.row(3) = sumatoriaGaussi.error.head(errorMax);
        matrizATexto(error, "error");
    
    matrizATexto(error, "error");
    Eigen::VectorXd tiempo(5);
    tiempo[0] = jaconi.tiempo;
    tiempo[1] = sumJacobi.tiempo;
    tiempo[2] = gaussi.tiempo;
    tiempo[3] = sumatoriaGaussi.tiempo;
    tiempo[4] = seconds;
    vectorATexto(tiempo, "tiempo");
    Eigen::VectorXd iter(4);
    iter[0] = jaconi.iter;
    iter[1] = sumJacobi.iter;
    iter[2] = gaussi.iter;
    iter[3] = sumatoriaGaussi.iter;
    vectorATexto(iter, "iteraciones");
    Eigen::VectorXd converge(4);
    converge[0] = jaconi.converge;
    converge[1] = sumJacobi.converge;
    converge[2] = gaussi.converge;
    converge[3] = sumatoriaGaussi.converge;
    vectorATexto(converge, "converge");
}
void printearTupla(Tupla iterativo, Eigen::MatrixXd A){
    std::cout << "Iteraciones: " << iterativo.iter << std::endl;
    std::cout << "Tiempo: " << iterativo.tiempo << std::endl;
    std::cout << "Converge: " << iterativo.converge << std::endl;
    //std::cout << "Error: " << iterativo.error << std::endl;
    std::cout << "X: \n" << iterativo.x << std::endl;
    std::cout << "Ax: \n " << A*iterativo.x << std::endl;
}

int main(int argc, char** argv){
    if (argc < 3) {
        std::cerr << "Falta un argumento: " << argv[0] << " <filename>  <filesol>" << std::endl;
        return 1;
    }   

    std::string filename =  std::string(argv[1]); 
    std::string filesol = std::string(argv[2]);
    std::string filenameSol = "";
    bool testeo = false;
    if (argc > 3){
        if (argc < 5){
            std::cerr << "Falta un argumento: " << argv[0] << " <filename>  <filesol> <testeo>" << std::endl;
            return 1;
        }
        if (std::string(argv[3]) == "testeo"){
            testeo = true;
        }
         filenameSol = std::string(argv[4]);
    }
    Eigen::MatrixXd A = leerMatrix(filename);
    Eigen::VectorXd b = leerSolucion(filesol);
    Eigen::VectorXd sol = Eigen::VectorXd::Zero(b.size());
    if (testeo){
        sol = leerSolucion(filenameSol);
    }

    try {
        A.inverse();
    } catch (const std::exception& e) {
        std::cout << "No se puede invertir" << std::endl;
    }
    int iter = 1000;
    double tol = 0.000001;
    Tupla jaconi = jacobi(A,b,sol, iter,tol, testeo);
    Tupla sumJacobi = sumatoriaJacobi(A,b, sol, iter,tol,  testeo);
    Tupla gaussi = gauus(A,b, sol, iter,tol,  testeo);
    Tupla sumatoriaGaussi = sumgauss(A,b,sol, iter,tol,  testeo );
    // Usamos un metoodo directo para comparar, LU de Eigen
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd x = A.lu().solve(b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();
    guardarResultados(jaconi, sumJacobi, gaussi, sumatoriaGaussi, x, seconds, testeo);
 
    return 0;
}
