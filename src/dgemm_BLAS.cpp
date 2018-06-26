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


void dgemmBLAS(const int&dim, const double* matrix_A, const double* matrix_B, double *matrix_C){
	         //openblas_set_num_threads(3);

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, matrix_A, dim, matrix_B, dim, 1.0, matrix_C, dim);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	std::cout << "BLAS ";
	//printDuration(t1, t2, dim);
}


