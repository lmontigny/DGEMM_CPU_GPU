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

void dgemmBLAS(const int&dim, const double* matrix_A, const double* matrix_B, double *matrix_C){
	         //openblas_set_num_threads(3);

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, matrix_A, dim, matrix_B, dim, 1.0, matrix_C, dim);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	std::cout << "BLAS ";
	printDuration(t1, t2, dim);
}

void dgemmForLoop(const int&dim, const double* matrix_A, const double* matrix_B, double *matrix_C){
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

	std::cout << "For Loop ";
	printDuration(t1, t2, dim);
}

void dgemmForLoopOpenMP(const int&dim, const double* matrix_A, const double* matrix_B, double *matrix_C){
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	#pragma omp parallel
	{
	int i,j,k;
	#pragma omp for
	for( i = 0; i < dim; i++ )
	     for( j = 0; j < dim; j++ )
	     {
	          double cij = matrix_C[i+j*dim];
	          for( k = 0; k < dim; k++ )
	      	 cij += matrix_A[i+k*dim] * matrix_B[k+j*dim];
	          matrix_C[i+j*dim] = cij;
	     }
}
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	std::cout << "For Loop openMP ";
	printDuration(t1, t2, dim);
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


	//dgemmForLoop(dim, matrix_A, matrix_B, matrix_C);
	//dgemmBLAS(dim, matrix_A, matrix_B, matrix_C);
	//dgemmForLoopOpenMP(dim, matrix_A, matrix_B, matrix_C);


	//printMatrix(dim, matrix_C);

	free(matrix_A);
	free(matrix_B);
	free(matrix_C);

	return 0;
}


