#ifndef DGEMM_OPENMP_H
#define DGEMM_OPENMP_H
 
void dgemmForLoopOpenMP(const int&dim, const double* matrix_A, const double* matrix_B, double *matrix_C);
 
 #endif
