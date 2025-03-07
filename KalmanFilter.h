//
// Created by amir on 2/27/2025.
//

#ifndef KALMANFILTER_H
#define KALMANFILTER_H


struct kalman_system {
    int x_num;
    int z_num;
    float** x;  // system state vector [x_num][1]
    float** z;  // measurements vector [z_num][1]
    float** A;  // state transition matrix [x_num][x_num]
    float** P;  // error covariance matrix [x_num][x_num]
    float** H;  // measurement matrix [z_num][x_num]
    float** Ht; // H transpose
    /*
      note: I don't need to store H transpose but in order
      to avoid extra loops I store it.
      important!!!: It is assumed the system is linear
    */
    float** Q;  // process noise covariance [x_num][x_num]
    float** R;  // measurement noise covariance [z_num][z_num]
};


kalman_system* init_kalman_system(int x_num, int z_num);
void kalman_perform(kalman_system* system);

#endif //KALMANFILTER_H
