import numpy as np
import matplotlib.pyplot as plt


class TimeDependentKalmanFilter(object):
    def __init__(self, a_x, a_y, std_acc, x_std_meas, y_std_meas):
        """
        :param a_x: acceleration in x-direction
        :param a_y: acceleration in y-direction
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        """

        # Define the  control input variables
        self.u = np.matrix([[a_x], [a_y]])

        # Intial State
        self.x = np.matrix([[0], [0], [0], [0]])

        # Define the State Transition Matrix A
        self.A = lambda dt: np.matrix([[1, 0, dt, 0],
                                       [0, 1, 0, dt],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])

        # Define the Control Input Matrix B
        self.B = lambda dt: np.matrix([[(dt ** 2) / 2, 0],
                                       [0, (dt ** 2) / 2],
                                       [dt, 0],
                                       [0, dt]])

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Initial Process Noise Covariance
        self.Q = lambda dt: np.matrix([[(dt ** 4) / 4, 0, (dt ** 3) / 2, 0],
                                       [0, (dt ** 4) / 4, 0, (dt ** 3) / 2],
                                       [(dt ** 3) / 2, 0, dt ** 2, 0],
                                       [0, (dt ** 3) / 2, 0, dt ** 2]]) * std_acc ** 2

        # Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas ** 2, 0],
                            [0, y_std_meas ** 2]])

        # Initial Covariance Matrix
        self.P = np.eye(self.A(0).shape[1])

    def predict(self, dt):
        # x_k =Ax_(k-1) + Bu_(k-1)
        self.x = np.dot(self.A(dt), self.x) + np.dot(self.B(dt), self.u)

        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(np.dot(self.A(dt), self.P), self.A(dt).T) + self.Q(dt)
        return self.x[0:2]

    def update(self, z):
        if z is None:
            return self.x[0:2]
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # K = P * H'* inv(S)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))

        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P

        return self.x[0:2]
