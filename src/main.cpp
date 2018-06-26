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
#include <omp.h>

#include "dgemm_openMP.h"
#include "dgemm_ForLoopNaive.h"
#include "dgemm_BLAS.h"



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

template<typename Clock>
void printDuration(const std::chrono::time_point<Clock> &t1, const std::chrono::time_point<Clock> &t2, const int &dim){
	auto duration = duration_cast<microseconds>( t2 - t1 ).count();
	std::cout << "duration: " <<  duration*1.0e-6 << std::endl;

	//double gflops = 2.0 * dim * dim * dim;
	//gflops = gflops/duration*1.0e-6;
	//std::cout << "GFLOPS: " << gflops << std::endl;
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
	std::cout <<"Number of elements: " << dim*dim << std::endl;
	std::cout << "Size of 1 Matrix : " << dim*dim*sizeof(double)*3/1024/1024 << " MB"<< std::endl;

	#pragma omp parallel
	for(int i=0; i < dim*dim; i++){
		matrix_A[i]=doubleRandomGenerator();
		matrix_B[i]=doubleRandomGenerator();
	}

//#ifdef _OPENMP
	std::cout << "Will use: "<< omp_get_max_threads() << " threads"<<std::endl;
	std::cout << "Will use: "<< omp_get_num_procs() << " threads"<<std::endl;
//#endif

	dgemmForLoopNaive(dim, matrix_A, matrix_B, matrix_C);
	dgemmBLAS(dim, matrix_A, matrix_B, matrix_C);
	dgemmForLoopOpenMP(dim, matrix_A, matrix_B, matrix_C);


	//printMatrix(dim, matrix_C);

	free(matrix_A);
	free(matrix_B);
	free(matrix_C);

	return 0;
}


