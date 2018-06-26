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

void dgemmForLoopNaive(const int&dim, const double* matrix_A, const double* matrix_B, double *matrix_C){
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
	//printDuration(t1, t2, dim);
}
