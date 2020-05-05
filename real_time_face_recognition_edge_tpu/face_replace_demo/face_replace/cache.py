# -*- coding: utf-8 -*-

import random

from scipy.spatial import cKDTree


class Cache:
    """
    Represents a cache used to store detected faces.
    Each entry consists of:
        The x1,x2,y1,y2 coordinates of the bounding box of the detected face, 
        The image used to replace the face (randomly chosen), 
        The age of the entry (cache entries expire)  
    """

    MAX_BBOX_DISTANCE = 50
    MAX_AGE = 10
    INITIAL_AGE = 0
    AGING = 5
    REJUVENATE = 1

    def __init__(self, face_filters):
        self.entries = []
        self.available_face_filters = face_filters

    def invalidate(self):
        """
        Invalidates the cache.
        Removes expired entries depending on there age. 
        """
        aged_entries = [
            [entry[0], entry[1], entry[2] + Cache.AGING] for entry in self.entries
        ]
        self.entries = [entry for entry in aged_entries if entry[2] < Cache.MAX_AGE]

    def update(self, bbox):
        """
        Updates the cache.
        Updates an existing entry if the bounding box of the new entry 
        is close enough otherwise creates a new entry.
        Returns the face filter of the new or updated entry. 
        """
        if len(self.entries) == 0:
            return self._add_new_entry(bbox)
        nearest_bbox_distance, nearest_bbox_index = self._nearest_bounding_box(bbox)
        if nearest_bbox_distance > Cache.MAX_BBOX_DISTANCE:
            return self._add_new_entry(bbox)
        return self._update_entry(bbox, nearest_bbox_index)

    def _add_new_entry(self, bbox):
        """Adds new cache entry and returns its randomly chosen face filter"""
        face_filter = random.choice(self.available_face_filters)
        self.entries.append([bbox, face_filter, Cache.INITIAL_AGE])
        return face_filter

    def _nearest_bounding_box(self, bbox):
        """
        Using nearest neighbor algorithm to return the nearest 
        bounding box for the given bounding box and its index
        """
        bb_matrix = [entry[0] for entry in self.entries]
        nearest_bbox_distance, nearest_bbox_index = cKDTree(bb_matrix).query(bbox, k=1)
        return nearest_bbox_distance, nearest_bbox_index

    def _update_entry(self, bbox, nearest_bbox_index):
        """
        Updates the bounding box of an existing cache entry, 
        decreases its age and returns its face filter
        """
        self.entries[nearest_bbox_index][0] = bbox
        self.entries[nearest_bbox_index][2] -= Cache.REJUVENATE
        return self.entries[nearest_bbox_index][1]
