import copy
import datetime
from threading import Thread, Lock
import cv2

from YOLOCell import YOLOData, detect_with_yolo
from config import SHOW_IMAGE_SIZE


class LockingDetectionThread(Thread):
    value = None
    delta_time = 0
    lock = Lock()

    def __init__(self, frame):
        super().__init__()
        self.frame = copy.deepcopy(frame)

    @staticmethod
    def with_lock(block):
        with LockingDetectionThread.lock:
            return block()

    def run(self):
        start_time = datetime.datetime.now()
        result = detect_with_yolo(self.frame)
        with self.lock:
            LockingDetectionThread.value = result
            finish_time = datetime.datetime.now()
            LockingDetectionThread.delta_time = round((finish_time - start_time).microseconds / 1000)


def copy_current_value():
    return copy.deepcopy(LockingDetectionThread.value)


def copy_delta_time():
    return copy.deepcopy(LockingDetectionThread.delta_time)


def show_result(frame, alpha=(1, 1)):
    result = LockingDetectionThread.with_lock(copy_current_value)
    delta_time = LockingDetectionThread.with_lock(copy_delta_time)

    if not (result is None):
        result_center_on_frame = (int(result.center_x * alpha[0]), int(result.center_y * alpha[1]))
        result_size_on_frame = (int(result.width * alpha[0]), int(result.height * alpha[1]))
        left_top = (result_center_on_frame[0] - result_size_on_frame[0] // 2,
                    result_center_on_frame[1] - result_size_on_frame[1] // 2)
        right_bottom = (result_center_on_frame[0] + result_size_on_frame[0] // 2,
                        result_center_on_frame[1] + result_size_on_frame[1] // 2)

        cv2.rectangle(frame, left_top, right_bottom, (0, 0, 255), 3)
        cv2.putText(frame, f"cls: {result.cls} with confidence {result.confidence}",
                    (left_top[0], left_top[1] - 10), 0, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"time spent {delta_time}ms",
                    (left_top[0], left_top[1] + 10), 0, 0.5, (255, 255, 255), 2)


def main_with_non_stable_detection():
    cap = cv2.VideoCapture(0)
    thread = None
    if cap.isOpened():
        while True:
            check, frame = cap.read()
            if check:
                if thread is None or not thread.is_alive():
                    thread = LockingDetectionThread(frame)
                    thread.start()
                resized = cv2.resize(frame, SHOW_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                show_result(resized, (SHOW_IMAGE_SIZE[0] / YOLO_IMAGE_SIZE[0], SHOW_IMAGE_SIZE[1] / YOLO_IMAGE_SIZE[1]))
                cv2.imshow('Resized frame', resized)

                key = cv2.waitKey(50)
                if key == ord('q'):
                    break
            else:
                print('Frame not available')
                print(cap.isOpened())


if __name__ == "__main__":
    # execute main
    main_with_non_stable_detection()
