# Purifier
Purifier is a Kalman Filter implementation for different systems.

The code is optimized for the Arduino boards, and due to resource limitations, custom algebraic functions are implemented.


# How to use
- Call `init_kalman_system` to create the system object.
- Update the values continuously and call `kalman_perform`.

Refer to the example for the IMU model to see its usage in practice.  