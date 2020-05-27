# -*- coding: utf-8 -*-

import sys

import cv2


def _exit_app(exit_code, video_stream):
    """Clean up and exit app"""
    cv2.destroyAllWindows()
    video_stream.stop()
    sys.exit(exit_code)


def _handle_right_click(event, _x, _y, _flags, video_stream):
    """Exits app on right click"""
    if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONDBLCLK):
        _exit_app(0, video_stream)


def show_frame(frame, video_stream, window_name):
    """
    Displays the given frame in a window with given 
    name and attache listener for exiting the app
    """
    # make videostream fullscreen
    cv2.namedWindow(window_name, cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # show the output frame and wait for a key press
    cv2.imshow(window_name, frame)
    # enable right click and 'q' to stop application
    cv2.setMouseCallback(window_name, _handle_right_click, video_stream)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        _exit_app(0, video_stream)
