# -*- coding: utf-8 -*-

import cv2
import imutils
from PIL import Image

IMAGE_WIDTH = 720


def preprocess(frame):
    """
    Resize frame and convert colors to RGB.
    Returns 
        a converted and resized copy as RGB PIL image, 
        a only resized copy as np array 
    of the given frame 
    """
    resized_frame = imutils.resize(frame, width=IMAGE_WIDTH)
    rgb_array = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    frame_as_image = Image.fromarray(rgb_array)
    return frame_as_image, resized_frame


def replace_face(bounding_box, frame, face_filter):
    """
    Draw the given replace image on the given frame inside the given bounding box. 
    Therefore resizes the image to fit inside the bounding box.
    """
    (bbox_x1, bbox_y1, bbox_x2, bbox_y2) = bounding_box
    width = bbox_x2 - bbox_x1
    height = bbox_y2 - bbox_y1
    face_filter_resized = cv2.resize(
        face_filter, (width, height), interpolation=cv2.INTER_AREA
    )
    return _override_image(
        face_filter_resized, frame, bbox_y1, bbox_y2, bbox_x1, bbox_x2
    )


def _override_image(face_filter, frame, bbox_y1, bbox_y2, bbox_x1, bbox_x2):
    """
    Overrides the frame with the replace image inside the bounding box.
    Since we are working with transparent png images, we consider the alpha value 
    of the replace image and draw the original pixel where the replace image is transparent
    """
    face_filter_alpha = face_filter[:, :, 3] / 255.0
    inverted_alpha = 1.0 - face_filter_alpha
    for colour_index in range(0, 3):
        frame[bbox_y1:bbox_y2, bbox_x1:bbox_x2, colour_index] = (
            face_filter_alpha * face_filter[:, :, colour_index]
            + inverted_alpha * frame[bbox_y1:bbox_y2, bbox_x1:bbox_x2, colour_index]
        )
    return frame
