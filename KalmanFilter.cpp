//
// Created by amir on 2/27/2025.
//

#include "KalmanFilter.h"
#include "Algebra.h"



kalman_system* init_kalman_system(int const x_num, int const z_num){

  kalman_system* sys = new kalman_system();
  sys -> x_num = x_num;
  sys -> z_num = z_num;

  sys -> x = init_matrix(x_num, 1);
  sys -> z = init_matrix(z_num, 1);

  sys -> A = init_matrix(x_num, x_num);
  sys -> P = init_matrix(x_num, x_num);
  sys -> H = init_matrix(z_num, x_num);
  sys -> Q = init_matrix(x_num, x_num);
  sys -> R = init_matrix(z_num, z_num);

  return sys;
}

// updates the pointer
void kalman_perform(kalman_system* system){

  // Prediction of system state
  float** xp = matrix_multiply(
    system -> A,
    system -> x,
    system -> x_num,
    system -> x_num,
    system -> x_num,
    1
  );

  float** A_P = matrix_multiply(
    system -> A,
    system -> P,
    system -> x_num,
    system -> x_num,
    system -> x_num,
    system -> x_num
  );

  float** A_transpose = transpose_matrix(
    system -> A,
    system -> x_num,
    system -> x_num
  );

  float** A_P_At = matrix_multiply(
    A_P,
    A_transpose,
    system -> x_num,
    system -> x_num,
    system -> x_num,
    system -> x_num
  );

  free_matrix(A_P, system -> x_num); // freeing memory

  // Prediction of error covariance
  float** Pp = matrix_add(
    A_P_At,
    system -> Q,
    system -> x_num,
    system -> x_num
  );

  free_matrix(A_P_At, system -> x_num); // freeing memory

  float** H_Pp = matrix_multiply(
    system -> H,
    Pp,
    system -> z_num,
    system -> x_num,
    system -> x_num,
    system -> x_num
  );

  float** H_Pp_Ht = matrix_multiply(
    H_Pp,
    system -> Ht,
    system -> z_num,
    system -> x_num,
    system -> x_num,
    system -> z_num
  );

  free_matrix(H_Pp, system -> z_num); // freeing memory

  float** H_Pp_Ht_plus_R = matrix_add(
    H_Pp_Ht,
    system -> R,
    system -> z_num,
    system -> z_num
  );

  free_matrix(H_Pp_Ht, system -> z_num); // freeing memory

  float** inv_H_Pp_Ht_plus_R = inverse_matrix(
    H_Pp_Ht_plus_R,
    system -> z_num
  );

  free_matrix(H_Pp_Ht_plus_R, system -> z_num); // freeing memory

  float** Pp_Ht = matrix_multiply(
    Pp,
    system -> Ht,
    system -> x_num,
    system -> x_num,
    system -> x_num,
    system -> z_num
  );

  // Kalman matrix
  float** K = matrix_multiply(
    Pp_Ht,
    inv_H_Pp_Ht_plus_R,
    system -> x_num,
    system -> z_num,
    system -> z_num,
    system -> z_num
  );

  free_matrix(Pp_Ht, system -> x_num); // freeing memory
  free_matrix(inv_H_Pp_Ht_plus_R, system -> z_num); // freeing memory

  float** K_z = matrix_multiply(
    K,
    system -> z,
    system -> x_num,
    system -> z_num,
    system -> z_num,
    1
  );

  float** K_H = matrix_multiply(
    K,
    system -> H,
    system -> x_num,
    system -> z_num,
    system -> z_num,
    system -> x_num
  );

  float** K_H_xp = matrix_multiply(
    K_H,
    xp,
    system -> x_num,
    system -> x_num,
    system -> x_num,
    1
  );

  float** K_z_sub_K_H_xp = matrix_sub(
    K_z,
    K_H_xp,
    system -> x_num,
    1
  );

  free_matrix(K_H_xp, system -> x_num); // freeing memory
  free_matrix(K_z, system -> x_num); // freeing memory

  // new system state
  system -> x = matrix_add(
    xp,
    K_z_sub_K_H_xp,
    system -> x_num,
    1
  );

  free_matrix(K_z_sub_K_H_xp, system -> x_num); // freeing memory

  float** K_H_Pp = matrix_multiply(
    K_H,
    Pp,
    system -> x_num,
    system -> x_num,
    system -> x_num,
    system -> x_num
  );

  free_matrix(K_H, system -> x_num);

  system -> P = matrix_sub(
    Pp,
    K_H_Pp,
    system -> x_num,
    system -> x_num
  );

  free_matrix(K_H_Pp, system -> x_num);
  free_matrix(Pp, system -> x_num);
  free_matrix(xp, system -> x_num);

}