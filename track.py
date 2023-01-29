#!\bin\python2.7

from kalman_filter import KalmanFilter
from tracker_utils import xxyy_to_xysr, xysr_to_xxyy
import numpy as np

import config

class KalmanTrack:
    """
    This class represents the internal state of individual tracked objects observed as bounding boxes.
    """
    def __init__(self, initial_state):
        """
        Initialises a tracked object according to initial bounding box.
        :param initial_state: (array) single detection in form of bounding box [X_min, Y_min, X_max, Y_max]
        """
        self.debug_flag = False#config.DEBUG_FLAG
        if self.debug_flag:
            print("KalmanTrack::init, initial_state: {}".format(initial_state))

        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # Transition matrix
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]])

        # Transformation matrix (Observation to State)
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.  # observation error covariance
        self.kf.P[4:, 4:] *= 1000.  # initial velocity error covariance
        self.kf.P *= 10.  # initial location error covariance
        self.kf.Q[-1, -1] *= 0.01  # process noise
        self.kf.Q[4:, 4:] *= 0.01  # process noise
        self.kf.x[:4] = xxyy_to_xysr(initial_state)  # initialize KalmanFilter state

        if self.debug_flag:
            print("KalmanTrack::init, kf.x.shape: {}".format(self.kf.x))
            print("KalmanTrack::init, kf.x[:4] {}".format(self.kf.x[:4]))

    def project(self):
        """
        :return: (ndarray) The KalmanFilter estimated object state in [x1,x2,y1,y2] format
        """
        if self.debug_flag:
            print("KalmanTrack::project")
        return xysr_to_xxyy(self.kf.x)

    def update(self, new_detection):
        """
        Updates track with new observation and returns itself after update
        :param new_detection: (np.ndarray) new observation in format [x1,x2,y1,y2]
        :return: KalmanTrack object class (itself after update)
        """
        if self.debug_flag:
            print("KalmanTrack::update")
        self.kf.update(xxyy_to_xysr(new_detection))
        return self

    def predict(self):
        """
        :return: ndarray) The KalmanFilter estimated new object state in [x1,x2,y1,y2] format
        """
        if self.debug_flag:
            print("KalmanTrack::predict")

        #if self.kf.x[6] + self.kf.x[2] <= 0:
        #    self.kf.x[6] *= 0.0
        if self.kf.x[7] + self.kf.x[2] <= 0:
            self.kf.x[7] *= 0.0
        self.kf.predict()

        return self.project()

