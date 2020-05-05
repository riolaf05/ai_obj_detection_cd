# -*- coding: utf-8 -*-

import glob
import os
import time

import cv2
from imutils.video import VideoStream
from edgetpu.detection.engine import DetectionEngine

import face_replace.app as app
import face_replace.frame_processor as frame_processor
from face_replace.cache import Cache


FACE_DETECTION_MODEL_PATH = os.path.normpath(
    os.path.join(
        os.getcwd(),
        "mobilenet_ssd_v2/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite",
    )
)
FACE_FILTER_PATHS = glob.glob("images/smileys/*.png")

WINDOW_NAME = "Face replace Demo"
MAX_FACES = 6


def run_video_loop(video_stream, face_filters, model_faces, confidence=0.3):
    """Runs the video loop which does the face replacement for each frame"""
    cache = Cache(face_filters)
    frames_counter = 0
    while True:
        frames_counter += 1
        input_frame = video_stream.read()
        processed_frame, frame = frame_processor.preprocess(cv2.flip(input_frame, 1))
        detected_faces = model_faces.detect_with_image(
            processed_frame,
            threshold=confidence,
            keep_aspect_ratio=True,
            relative_coord=False,
            top_k=MAX_FACES,
        )
        for face in detected_faces:
            bounding_box = face.bounding_box.flatten().astype("int")
            face_filter = cache.update(bounding_box)
            frame = frame_processor.replace_face(
                bounding_box, frame, face_filter
            )
        if len(cache.entries) > 0 and frames_counter % 10 == 0:
            cache.invalidate()
        app.show_frame(frame, video_stream, WINDOW_NAME)


def initialize():
    """
    Initializes the application. 
    Loads the face replavailable_face_filtersts the video stream.
    """
    face_filters = [cv2.imread(path, -1) for path in FACE_FILTER_PATHS]
    model_faces = DetectionEngine(FACE_DETECTION_MODEL_PATH)
    # initialize the video stream and allow the camera sensor to warmup
    video_stream = VideoStream(src=0).start()
    time.sleep(1.0)
    return (video_stream, face_filters, model_faces)


if __name__ == "__main__":
    video_stream, face_filters, model_faces = initialize()
    run_video_loop(video_stream, face_filters, model_faces)
