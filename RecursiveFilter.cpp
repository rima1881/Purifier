//
// Created by amir on 3/2/2025.
//

#include "RecursiveFilter.h"

float average(const float x) {
    static float x_bar = 0;
    static int k = 0;
    const float alpha = 1.0 - 1.0/++k;
    x_bar = alpha * x_bar + (1 - alpha) * x;
    return x_bar;
}

float moving_average(const float x) {
    static float x_bar = 0;
    static int k = 0;
    static float x_n = 0;
}