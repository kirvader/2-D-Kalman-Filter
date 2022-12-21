from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from config import CONFIDENCE_THRESHOLD, MODEL_PATH_32, MODEL_PATH_64, MODEL_PATH_128, MODEL_PATH_256, MODEL_PATH_512


class ORTSessions:
    ort_sess_32 = None
    ort_sess_64 = None
    ort_sess_128 = None
    ort_sess_256 = None
    ort_sess_512 = None

    @staticmethod
    def init():
        project_root = Path().resolve() / '..'

        if ORTSessions.ort_sess_32 is None:
            ORTSessions.ort_sess_32 = ort.InferenceSession(str(project_root / MODEL_PATH_32))
        if ORTSessions.ort_sess_64 is None:
            ORTSessions.ort_sess_64 = ort.InferenceSession(str(project_root / MODEL_PATH_64))
        if ORTSessions.ort_sess_128 is None:
            ORTSessions.ort_sess_128 = ort.InferenceSession(str(project_root / MODEL_PATH_128))
        if ORTSessions.ort_sess_256 is None:
            ORTSessions.ort_sess_256 = ort.InferenceSession(str(project_root / MODEL_PATH_256))
        if ORTSessions.ort_sess_512 is None:
            ORTSessions.ort_sess_512 = ort.InferenceSession(str(project_root / MODEL_PATH_512))

    @staticmethod
    def choose_model_by_detection_zone_sides(detection_width, detection_height):
        measure = (detection_width + detection_height) / 2
        project_root = Path().resolve() / '..'
        if measure < 64:
            return ort.InferenceSession(str(project_root / MODEL_PATH_32)), 32
        if measure < 128:
            return ort.InferenceSession(str(project_root / MODEL_PATH_32)), 32
        if measure < 256:
            return ort.InferenceSession(str(project_root / MODEL_PATH_64)), 64
        if measure < 512:
            return ort.InferenceSession(str(project_root / MODEL_PATH_128)), 128
        return ort.InferenceSession(str(project_root / MODEL_PATH_256)), 256


class YOLOData:
    def __init__(self, center_x, center_y, width, height, cls, confidence):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.cls = cls
        self.confidence = confidence

    def __str__(self):
        return f"({self.center_x}, {self.center_y}), ({self.width}, {self.height}), {self.cls}, {self.confidence}"

    @staticmethod
    def from_yolo_list(yolo_list: list):
        center_x = yolo_list[0]
        center_y = yolo_list[1]
        width = yolo_list[2]
        height = yolo_list[3]
        confidence = yolo_list[4]
        scores = np.array(yolo_list[5:])
        max_score_class = scores.argmax()
        return YOLOData(center_x, center_y, width, height, max_score_class, confidence * scores[max_score_class])


def prepare_data_and_determine_model_to_detect(frame, left, top, right, bottom):
    shape = frame.shape
    height = shape[0]
    width = shape[1]
    detection_width = (right - left) * width
    detection_height = (bottom - top) * height

    model_session, model_size = ORTSessions.choose_model_by_detection_zone_sides(detection_width, detection_height)

    new_image_size = (round(model_size * width / detection_width), round(model_size * height / detection_height))

    resized_frame = np.copy(cv2.resize(frame, new_image_size, interpolation=cv2.INTER_AREA))

    height = new_image_size[0]
    width = new_image_size[1]
    left_on_image = min(max(round(left * width), 0), width - model_size)
    top_on_image = min(max(round(top * height), 0), height - model_size)
    right_on_image = left_on_image + model_size
    bottom_on_image = top_on_image + model_size

    result = [[[] for _ in range(model_size)],
              [[] for _ in range(model_size)],
              [[] for _ in range(model_size)]]
    for i in range(top_on_image, bottom_on_image):
        for j in range(left_on_image, right_on_image):
            for k in range(resized_frame.shape[2]):
                color_value = resized_frame[i][j][k] / 255.0
                result[k][i - top_on_image].append(color_value)
    return [result], model_session, model_size


def fetch_yolo_most_relevant_value_by_class(output, cls):
    result = None
    for yolo_list in output:
        current = YOLOData.from_yolo_list(yolo_list)
        if current.cls == cls and current.confidence > CONFIDENCE_THRESHOLD:
            if result is None:
                result = current
            elif current.confidence > result.confidence:
                result = current
    return result


def process_image(data, ort_session, model_side_size):
    output = ort_session.run(None, {'images': data})[0][0]
    result = fetch_yolo_most_relevant_value_by_class(output, 0)
    if result is None:
        return None
    new_result = YOLOData(
        result.center_x / model_side_size,
        result.center_y / model_side_size,
        result.width / model_side_size,
        result.height / model_side_size,
        result.cls,
        result.confidence
    )
    return new_result


def detect_with_yolo(frame, left=0, top=0, right=1, bottom=1):
    prepared_data, session, model_side_size = prepare_data_and_determine_model_to_detect(frame,
                                                                                         left,
                                                                                         top,
                                                                                         right,
                                                                                         bottom)
    return process_image(prepared_data, session, model_side_size)
