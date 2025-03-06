#include <iostream>
#include "Algebra.h"
#include "RecursiveFilter.h"

int main() {

    float** matrix = init_matrix(4, 4);

    matrix[0][0] = 4;
    matrix[0][1] = 3;
    matrix[0][2] = 9;
    matrix[0][3] = 2;

    matrix[1][0] = 8;
    matrix[1][1] = 0;
    matrix[1][2] = 3;
    matrix[1][3] = 4;

    matrix[2][0] = 5;
    matrix[2][1] = 1;
    matrix[2][2] = 6;
    matrix[2][3] = 3;

    matrix[3][0] = 7;
    matrix[3][1] = 5;
    matrix[3][2] = 2;
    matrix[3][3] = 1;

    print_matrix(matrix, 4, 4);


    std::cout << average(10) << std::endl;
    std::cout << average(20) << std::endl;
    std::cout << average(30) << std::endl;



    return 0;
}
