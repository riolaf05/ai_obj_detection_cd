# -*- coding: utf-8 -*-

import os
import unittest

import cv2
import numpy as np

from face_replace.cache import Cache


class TestCache(unittest.TestCase):
    def setUp(self):
        self.boxes = [(0, 0, 1, 1), (0, 0, 1.5, 1.5), (0, 0, 1000, 1000)]
        self.replace_images = ["img1", "img2", "img3"]
        self.class_under_test = Cache(self.replace_images)

    def test_update_adds_new_if_empty(self):
        expected_box = self.boxes[0]

        self.class_under_test.update(expected_box)

        self.assert_cache_entry(
            expected_len=1,
            entry_index=0,
            expected_box=expected_box,
            expected_age=Cache.INITIAL_AGE,
        )

    def test_update_replaces_existing_entry_and_reduces_age(self):
        expected_box = self.boxes[1]
        age = 10
        expected_age = age - 1
        self.class_under_test.entries.append([self.boxes[0], "img1", age])

        self.class_under_test.update(expected_box)

        self.assert_cache_entry(
            expected_len=1,
            entry_index=0,
            expected_box=expected_box,
            expected_age=expected_age,
        )

    def test_update_appends_new_entry(self):
        expected_box = self.boxes[2]
        expected_img = self.replace_images[2]
        self.class_under_test.entries.append(
            [self.boxes[0], self.replace_images[0], Cache.INITIAL_AGE]
        )

        self.class_under_test.update(expected_box)

        self.assert_cache_entry(
            expected_len=2,
            entry_index=1,
            expected_box=expected_box,
            expected_age=Cache.INITIAL_AGE,
        )

    def test_invalidate_updates_age(self):
        self.class_under_test.entries.append(
            ["box", self.replace_images[0], Cache.INITIAL_AGE]
        )

        self.class_under_test.invalidate()

        self.assert_cache_entry(
            expected_len=1,
            entry_index=0,
            expected_box="box",
            expected_age=Cache.INITIAL_AGE + Cache.AGING,
            expected_image=self.replace_images[0],
        )

    def test_invalidate_updates_age_two_entries(self):
        age1 = 2
        age2 = 3
        expected_age1 = age1 + Cache.AGING
        expected_age2 = age2 + Cache.AGING
        self.class_under_test.entries.append(["box1", self.replace_images[0], age1])
        self.class_under_test.entries.append(["box2", self.replace_images[1], age2])

        self.class_under_test.invalidate()

        self.assert_cache_entry(
            expected_len=2,
            entry_index=0,
            expected_box="box1",
            expected_age=expected_age1,
            expected_image=self.replace_images[0],
        )
        self.assert_cache_entry(
            expected_len=2,
            entry_index=1,
            expected_box="box2",
            expected_age=expected_age2,
            expected_image=self.replace_images[1],
        )

    def test_invalidate_removes_entries_if_expired(self):
        age = 4
        self.class_under_test.entries.append(["box1", self.replace_images[1], age])
        self.class_under_test.entries.append(["box2", self.replace_images[2], 6])

        self.class_under_test.invalidate()

        self.assert_cache_entry(
            expected_len=1,
            entry_index=0,
            expected_box="box1",
            expected_age=age + Cache.AGING,
            expected_image=self.replace_images[1],
        )

    def assert_cache_entry(
        self, expected_len, entry_index, expected_box, expected_age, expected_image=None
    ):
        cached = self.class_under_test.entries
        self.assertEqual(len(cached), expected_len)
        self.assertEqual(cached[entry_index][0], expected_box)
        if expected_image == None:
            self.assertTrue(cached[entry_index][1], any(self.replace_images))
        else:
            self.assertEqual(cached[entry_index][1], expected_image)
        self.assertEqual(cached[entry_index][2], expected_age)
