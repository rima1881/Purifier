cmake_minimum_required(VERSION 3.29)
project(Purifier)

set(CMAKE_CXX_STANDARD 11)

add_executable(Purifier main.cpp
        KalmanFilter.cpp
        KalmanFilter.h
        Algebra.cpp
        Algebra.h
)
