// Double General Matrix Multiplication (DGEMM)
// Author: Laurent Montigny
// Created: June 2018

#include <random>
#include <iostream>
#include <memory>
#include <cblas.h>
#include <chrono>

#include <sys/time.h>
#include <time.h>


using namespace std::chrono;



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

	dim = 512;
	matrix_A = (double *)malloc( dim*dim*sizeof( double ));
	matrix_B = (double *)malloc( dim*dim*sizeof( double ));
	matrix_C = (double *)malloc( dim*dim*sizeof( double ));


	std::cout <<"Size of double: " << sizeof(double) << std::endl;
	std::cout << "Size of 1 Matrix : " << dim*dim*sizeof(double)*3/1024/1024 << " MB"<< std::endl;

	for(int i=0; i < dim*dim; i++){
		matrix_A[i]=doubleRandomGenerator();
		matrix_B[i]=doubleRandomGenerator();
	}

high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for(int i=0; i<dim; i++){  // row
		for(int j=0; j<dim; j++){ // col
				double sum = 0;
				for(int k=0; k<dim; k++){
					sum += matrix_A[i*dim + k] * matrix_B[k*dim + j];
				}
				matrix_C[i*dim+j] = sum;
		}
	}
high_resolution_clock::time_point t2 = high_resolution_clock::now();
    //auto duration = duration_cast<microseconds>( t2 - t1 ).count();

    //std::cout << duration << std::endl;

	//printMatrix(dim, matrix_C);

  struct timeval start,finish;
  double duration;

gettimeofday(&start, NULL);
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, matrix_A, dim, matrix_B, dim, 1.0, matrix_C, dim);
  gettimeofday(&finish, NULL);

  duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
  double gflops = 2.0 * dim * dim * dim;
  gflops = gflops/duration*1.0e-6;

std::cout << gflops << " " << duration;



	free(matrix_A);
	free(matrix_B);
	free(matrix_C);

	return 0;
}


