import config
from kalman_filter_predict import KalmanFilterDetection

def demo():
    """ Starts the tracker on source video. Can start multiple instances of SORT in parallel """
    path_to_video = config.VIDEO_INPUT_FILE_NAME
    print("path_to_video: {}".format(path_to_video))
    mot_tracker = KalmanFilterDetection(path_to_video)


if __name__ == '__main__':
    demo()