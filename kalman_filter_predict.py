#!\bin\python2.7

"""
Main module for the real-time tracker class execution. Based on the SORT algorithm
"""

from __future__ import print_function
import os.path
import numpy as np
import cv2

from track import KalmanTrack
from object_detection_api import ObjectDetectorAPI
from netutils import ObjDetArchType
import config
import time

class KalmanFilterDetection:

    def __init__(self, src=None, detector='dyhead-fpn'):
        self.kalman_tracker = None
        if src is not None:
            self.src = cv2.VideoCapture(src)

        self.window_name = "Detection"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        width = int(self.src.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.src.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = self.src.get(cv2.CAP_PROP_FPS)
        #num_frames = int(self.src.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.src is not None:
            self.video_writer = cv2.VideoWriter(config.VIDEO_OUTPUT_FILE_NAME, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                           fps=float(frames_per_second),
                                           frameSize=(width, height), isColor=True)

        print("width: {}, height: {}".format(width, height))

        self.detector = None

        if detector == 'dyhead-fpn':
            self.detector = ObjectDetectorAPI(ObjDetArchType.DYHEAD_FPN)

        self.score_threshold = 0.7
        self.start_tracking()

    def close(self):
        self.src.release()
        self.video_writer.release()

    def next_frame(self):
        _, frame = self.src.read()
        #frame = cv2.resize(frame, (1280, 720))
        #print("frame{}".format(frame.shape))
        start = time.time()
        boxes, scores, classes, num = self.detector.processFrame(frame)
        end = time.time()
        elapsed_time = (end - start) * 1000
        print("Evaluation Time for Object Detection: {} ms ", elapsed_time)
        
        print("boxes: {}".format(boxes))
        return frame, boxes, elapsed_time

    def start_tracking(self):
        while True:
            # Fetch the next frame from video source, if no frames are fetched, stop loop
            frame, detections, detection_time = self.next_frame()
            #print("tracking main loop, frame:{}, detections:{}".format(frame, detections))
            if frame is None:
                break

            for bbox in detections:
                if self.kalman_tracker is None:
                    self.kalman_tracker = KalmanTrack(bbox)
                start = time.time()
                self.kalman_tracker.update(bbox)
                
                end = time.time()
                update_time = (end - start) * 1000
                print("Evaluation Time for Kalman Filter Update: {} ms ", update_time)  
                x1, y1, x2, y2 = [int(i) for i in bbox]
                #cv2.rectangle(frame,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),(255,0,0),2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                
                start1 = time.time()
                x3, y3, x4, y4 = self.kalman_tracker.predict()
                end1 = time.time()
                predict_time = (end1 - start1) * 1000
                print("Evaluation Time for Kalman Filter Predict: {} ms ", predict_time)
                print("Total Time Taken by Kalman Filter: {}".format(update_time + predict_time))
                print("x3 : {}, y3: {}".format(x3, y3))
                print("x4 : {}, y4: {}".format(x4, y4))
 
                cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), 2)
                text = "Total Time Taken by Kalman Filter with Red ROI {} ms".format(update_time + predict_time)
                cv2.putText(frame, text, (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 2)

                text = "Total Time Taken by ObjectDetectorAPI with Blue ROI {} ms".format(detection_time)
                cv2.putText(frame, text, (100, 150), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 2)
            # Show tracked frame
            cv2.imshow(self.window_name, frame)
            self.video_writer.write(frame)

            # if the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print('SORT operation terminated by user... closing tracker')
                return

    def verify_bbox_format(self, bbox):
        """
        Fixes bounding box format according to video type (e.g. benchmark test or video capture)
        :param bbox: (array) list of bounding boxes
        :return: (array) reformatted bounding box
        """
        if self.benchmark:
            return bbox.astype("int")
        else:
            bbox.astype("int")
            return [bbox[1], bbox[0], bbox[3], bbox[2]]


    @staticmethod
    def show_source(seq, frame, phase='train'):
        """ Method for displaying the origin video being tracked """
        return cv2.imread('mot_benchmark/%s/%s/img1/%06d.jpg' % (phase, seq, frame))

    @staticmethod
    def check_data_path():
        """ Validates correct implementation of symbolic link to data for SORT """
        if not os.path.exists('mot_benchmark'):
            print('''
            ERROR: mot_benchmark link not found!\n
            Create a symbolic link to the MOT benchmark\n
            (https://motchallenge.net/data/2D_MOT_2015/#download)
            ''')
            exit()


def main():
    """ Starts the tracker on source video. Can start multiple instances of SORT in parallel """
    path_to_video = config.VIDEO_INPUT_FILE_NAME
    print("path_to_video: {}".format(path_to_video))
    mot_tracker = KalmanFilterDetection(path_to_video)


if __name__ == '__main__':
    main()
