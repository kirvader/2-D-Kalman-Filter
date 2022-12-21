import copy
from time import time
from datetime import datetime

import cv2
from threading import Thread, Lock

import numpy as np

from YOLOObjectDetector import detect_with_yolo
from TimeDependentKalmanFilter import TimeDependentKalmanFilter
from yolo.GuessArea import guess_area
from yolo.YOLOCell import ORTSessions


class WorkerController:
    last_successful_detection_time = None
    last_frame_time = None
    last_detection_max_side_size = 1
    current_detection_zone = ((0, 0), (1, 1))
    current_detected_position = None
    estimate_position = (0, 0)

    lock = Lock()

    @staticmethod
    def init():
        if WorkerController.last_frame_time is None:
            WorkerController.last_frame_time = datetime.now()
        if WorkerController.last_successful_detection_time is None:
            WorkerController.last_successful_detection_time = WorkerController.last_frame_time

    @staticmethod
    def get_current_detection_zone():
        with WorkerController.lock:
            return WorkerController.current_detection_zone

    @staticmethod
    def get_current_detected_position():
        with WorkerController.lock:
            return WorkerController.current_detected_position

    @staticmethod
    def get_current_estimate_position():
        with WorkerController.lock:
            return WorkerController.estimate_position

    @staticmethod
    def get_last_frame_time():
        with WorkerController.lock:
            return WorkerController.last_frame_time

    @staticmethod
    def get_last_success_time():
        with WorkerController.lock:
            return WorkerController.last_successful_detection_time

    @staticmethod
    def get_last_success_max_side_size():
        with WorkerController.lock:
            return WorkerController.last_detection_max_side_size

    @staticmethod
    def set_current_detection_zone(zone):
        with WorkerController.lock:
            WorkerController.current_detection_zone = zone

    @staticmethod
    def set_current_detected_position(position):
        with WorkerController.lock:
            WorkerController.current_detected_position = position

    @staticmethod
    def set_current_estimate_position(position):
        with WorkerController.lock:
            WorkerController.estimate_position = position

    @staticmethod
    def set_last_frame_time(frame_time):
        with WorkerController.lock:
            WorkerController.last_frame_time = frame_time

    @staticmethod
    def set_last_success_time(frame_time):
        with WorkerController.lock:
            WorkerController.last_successful_detection_time = frame_time

    @staticmethod
    def set_last_success_max_side_size(max_side_size):
        with WorkerController.lock:
            WorkerController.last_detection_max_side_size = max_side_size


def relative_to_absolute(left, top, result, detection_width, detection_height):
    if result is None:
        return None
    return (left + result.center_x * detection_width, top + result.center_y * detection_height)


def make_state(abs_result):
    if abs_result is None:
        return None
    return np.array([[abs_result[0]], [abs_result[1]]])


def detect_cycle(frame, KF):
    current_time = datetime.now()

    # count delta time since last frame
    dt = (current_time - WorkerController.get_last_frame_time()).seconds

    # Predict next pos
    (x, y) = KF.predict(dt)

    dt_since_last_success = (current_time - WorkerController.get_last_success_time()).seconds

    # Guess area where we should look for object
    left, top, right, bottom = guess_area(x.item((0, 0)), y.item((0, 0)), WorkerController.last_detection_max_side_size,
                                          dt_since_last_success)

    WorkerController.set_current_detection_zone(((left, top), (right, bottom)))

    # Detect object
    relative_result = detect_with_yolo(frame, left, top, right, bottom)
    absolute_result = relative_to_absolute(left, top, relative_result, right - left, bottom - top)
    if not (absolute_result is None):
        WorkerController.set_last_success_time(current_time)
        WorkerController.set_last_success_max_side_size(min(relative_result.width, relative_result.height))
        WorkerController.set_current_detected_position(absolute_result)
    WorkerController.set_last_frame_time(current_time)

    # update kalman filter and show estimate
    estimate = KF.update(make_state(absolute_result))
    WorkerController.set_current_estimate_position((estimate.item((0, 0)), estimate.item((1, 0))))


class TrackingThread(Thread):
    delta_time = 0
    lock = Lock()

    def __init__(self, frame, KF):
        super().__init__()
        self.frame = copy.deepcopy(frame)
        self.kalman_filter = KF

    @staticmethod
    def with_lock(block):
        with TrackingThread.lock:
            return block()

    def run(self):
        start_time = time()
        detect_cycle(self.frame, self.kalman_filter)
        with self.lock:
            finish_time = time()
            TrackingThread.delta_time = round((finish_time - start_time) * 1000)
            # print(self.delta_time)


def copy_delta_time():
    return copy.deepcopy(TrackingThread.delta_time)


def show_results_on_frame(frame):
    detection_result = WorkerController.get_current_detected_position()
    detection_zone = WorkerController.get_current_detection_zone()
    estimate_result = WorkerController.get_current_estimate_position()
    delta_time = TrackingThread.with_lock(copy_delta_time)
    width = frame.shape[1]
    height = frame.shape[0]
    if not (detection_result is None):
        cv2.circle(frame, (round(detection_result[0] * width), round(detection_result[1] * height)), 10, (255, 0, 0),
                   -1)
        cv2.putText(frame, "Detected", (round(detection_result[0] * width), round(detection_result[1] * height + 10)), 0,
                    0.5, (255, 255, 255), 2)

    cv2.rectangle(frame, (round(detection_zone[0][0] * width), round(detection_zone[0][1] * height)),
                  (round(detection_zone[1][0] * width), round(detection_zone[1][1] * height)), (0, 255, 0), 3)
    cv2.putText(frame, "Current detection zone",
                (round(detection_zone[0][0] * width), round(detection_zone[0][1] * height)), 0, 0.5,
                (255, 255, 255), 2)

    cv2.circle(frame, (round(estimate_result[0] * width), round(estimate_result[1] * height)), 10, (0, 0, 255), -1)
    cv2.putText(frame, "Estimate", (round(estimate_result[0] * width), round(estimate_result[1] * height) - 10), 0, 0.5,
                (255, 255, 255), 2)

    cv2.putText(frame, f"time spent {delta_time}ms", (0, 10), 0, 0.5, (255, 255, 255), 2)


def main():
    VideoCap = cv2.VideoCapture(0)
    WorkerController.init()
    ORTSessions.init()

    KF = TimeDependentKalmanFilter(1, 1, 0.01, 0.01, 0.01)
    thread = None

    while True:
        # Read frame
        ret, frame = VideoCap.read()
        if not ret:
            break

        if thread is None or not thread.is_alive():
            thread = TrackingThread(frame, KF)
            thread.start()

        show_results_on_frame(frame)
        cv2.imshow("With tracker:", frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    # execute main
    main()
