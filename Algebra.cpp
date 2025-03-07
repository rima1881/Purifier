//
// Created by amir on 2/27/2025.
//

#include "Algebra.h"

#include <cmath>
#include <iostream>
#include <ostream>
#include <stdio.h>
#include <cmath>

float** init_eye(const int size){
    float** matrix = new float*[size];
    for(int i = 0; i < size; i++) {
        matrix[i] = new float[size]();
        matrix[i][i] = 1;
    }
    return matrix;
}

float** init_matrix(const int i, const int j) {
    float** matrix = new float*[i];
    for(int k=0; k<i; k++) matrix[k] = new float[j]();
    return matrix;
}

float** inverse_matrix(float** matrix, const int size) {

    float** augmented = init_matrix(size, size * 2);
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) augmented[i][j] = matrix[i][j];
        augmented[i][i + size] = 1;
    }

    for(int i=0; i < size; i++) {

        // Swapping row if item 0
        if(augmented[i][i] == 0.0) {
            for(int j = i + 1; j < size; j++) {
                // swap row j and i
                if (augmented[j][i] != 0.0) {
                    float* temp = augmented[j];
                    augmented[j] = augmented[i];
                    augmented[i] = temp;
                    break;
                }
            }
        }

        if(augmented[i][i] == 0.0) {
            free_matrix(augmented, size);
            return nullptr;
        }

        const float div = 1.0 / augmented[i][i];
        for(int j=0; j < size * 2; j++) augmented[i][j] = augmented[i][j] * div;

        for(int k=0; k < size; k++) {
            if(k == i)
                continue;
            const float temp = augmented[k][i];
            for(int l=0; l < size * 2; l++) {
                augmented[k][l] -= augmented[i][l] * temp;
            }
        }
    }

    print_matrix(augmented, size, size * 2);
    float** result = init_matrix(size, size);


    // TODO might be extra
    //preparing result and checking if the answer is correct
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {

            result[i][j] = augmented[i][j + size];

            // The algorithm always makes matrix[i][i] zero
            if(i == j)
                continue;

            // The matrix is not invertible
            if( std::fabs( augmented[i][j]) > 1e-6 ) {
                free_matrix(augmented, size);
                free_matrix(result, size);
                return nullptr;
            }
        }
    }

    free_matrix(augmented, size);
    return result;
}

void print_matrix(float** matrix, int rows, int cols) {

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%f , ", matrix[i][j]);
        }
        printf("\n");
    }
}

float** transpose_matrix(float** matrix, const int i, const int j) {
    float** transposed = new float*[j];
    for(int m=0; m<j; m++) {
        transposed[m] = new float[i];
        for(int n=0; n<i; n++) transposed[m][n] = matrix[n][m];
    }
    return transposed;
}

float** matrix_multiply(float**v1, float**v2, const int i1, const int j1, const int i2, const int j2) {

    if (j1 != i2) return nullptr;
    float** matrix = init_matrix(i1, j2);
    for (int i=0; i<i1; i++) {
        for (int j=0; j<j2; j++) {
            float result = 0;
            for (int k=0; k<j1; k++) result += v1[i][k] * v2[k][j];
            matrix[i][j] = result;
        }
    }
    return matrix;
}

void free_matrix(float** matrix, const int rows) {
    for (int i=0; i<rows; i++) delete[] matrix[i];
    delete[] matrix;
}


float** matrix_add(float** v1, float** v2, const int i, const int j){
    float** result = init_matrix(i, j);
    for(int x=0; x < i; x++)
        for(int y=0; y < j; y++)
            result[x][y] = v1[x][y] + v2[x][y];
    return result;
}

float** matrix_sub(float** v1, float** v2, const int i, const int j){
    float** result = init_matrix(i, j);
    for(int x=0; x < i; x++)
        for(int y=0; y < j; y++)
            result[x][y] = v1[x][y] - v2[x][y];
    return result;
}
