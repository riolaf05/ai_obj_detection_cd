# -*- coding: utf-8 -*-
import unittest
import cv2
import os
import face_replace.frame_processor as module_under_test
import numpy as np


class TestFrameProcessor(unittest.TestCase):
    def setUp(self):
        self.tmp_image_path = os.path.join(os.getcwd(), "images/test_resources/tmp.jpg")
        test_image_path = os.path.join(
            os.getcwd(), "images/test_resources/input_face_replace.jpg"
        )
        self.input_img = cv2.imread(test_image_path, -1)
        smiley_image_path = os.path.join(os.getcwd(), "images/smileys/smiley2.png")
        self.img_smiley = cv2.imread(smiley_image_path, -1)
        self.expected_image = cv2.imread(
            os.path.join(os.getcwd(), "images/test_resources/ref_face_replace.jpg"), -1
        )

    def test_preprocess_resizes_image(self):
        converted_and_resized, resized = module_under_test.preprocess(self.input_img)

        self.assertEqual(converted_and_resized.width, module_under_test.IMAGE_WIDTH)
        self.assertEqual(resized.shape[1], module_under_test.IMAGE_WIDTH)

    def test_preprocess_converts_image(self):
        converted_and_resized, _ = module_under_test.preprocess(self.input_img)

        self.assertEqual(converted_and_resized.mode, "RGB")

    def test_replace_face_replaces_faces(self):
        converted_and_resized, resized = module_under_test.preprocess(self.input_img)
        bounding_boxes = [
            [528, 199, 717, 427],
            [131, 178, 219, 281],
            [385, 196, 474, 299],
        ]

        for bounding_box in bounding_boxes:
            resized = module_under_test.replace_face(
                bounding_box, resized, self.img_smiley
            )

        cv2.imwrite(self.tmp_image_path, resized)
        loaded_tmp_image = cv2.imread(self.tmp_image_path, -1)
        self.assertTrue(image_equals(loaded_tmp_image, self.expected_image))

    def tearDown(self):
        if os.path.exists(self.tmp_image_path):
            os.remove(self.tmp_image_path)


def image_equals(image1, image2):
    return image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any())
