// Double General Matrix Multiplication (DGEMM)
// Author: Laurent Montigny
// Created: June 2018

#include <random>
#include <iostream>
#include <memory>

double doubleRandomGenerator(){
	std::random_device rd;  
	std::mt19937 gen(rd()); 
	std::uniform_real_distribution<> dis(0.0, 1.0);
	return dis(gen);
}

void printMatrix(const int &dim, const double *matrix){
	for(int i=0; i<dim; i++){  // row
		for(int j=0; j<dim; j++){ // col
				std::cout << matrix[i*dim+j] << std::endl;
		}
	}
}

int main(int argc, char const *argv[]){
	
	int dim;
	double *matrix_A, *matrix_B, *matrix_C;
	//std::unique_ptr<double> matrix_A = std::make_unique<double>();

	dim = 64;
	matrix_A = (double *)malloc( dim*dim*sizeof( double ));
	matrix_B = (double *)malloc( dim*dim*sizeof( double ));
	matrix_C = (double *)malloc( dim*dim*sizeof( double ));

	for(int i=0; i < dim*dim; i++){
		matrix_A[i]=doubleRandomGenerator();
		matrix_B[i]=doubleRandomGenerator();
	}

	for(int i=0; i<dim; i++){  // row
		for(int j=0; j<dim; j++){ // col
				double sum = 0;
				for(int k=0; k<dim; k++){
					sum += matrix_A[i*dim + k] * matrix_B[k*dim + j];
				}
				matrix_C[i*dim+j] = sum;
		}
	}

	printMatrix(dim, matrix_C);

	free(matrix_A);
	free(matrix_B);
	free(matrix_C);
}


