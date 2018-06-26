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
//       printDuration(t1, t2, dim);
}
	 

