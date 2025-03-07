//
// Created by amir on 2/27/2025.
//

#ifndef ALGEBRA_H
#define ALGEBRA_H

// IMPORTANT!!!
// Due to resources scarcity none of the given functions check for nullptr exception or bounding errors

float** init_eye(int size);

float** init_matrix(int i, int j);

// Function to perform Gaussian elimination to find the inverse
// IMPORTANT!!!
// returns nullptr if it doesn't have inverse
float** inverse_matrix(float** matrix, int n);

float** transpose_matrix(float** matrix, int i, int j);

void free_matrix(float** matrix, int rows);

void print_matrix(float** matrix, int rows, int cols);

// IMPORTANT!!!
// returns nullptr if j1 != i2
float** matrix_multiply(float**v1, float**v2, int i1, int j1, int i2, int j2);

float** matrix_add(float** v1, float** v2, int i, int j);

float** matrix_sub(float** v1, float** v2, int i, int j);

#endif // ALGEBRA_H
