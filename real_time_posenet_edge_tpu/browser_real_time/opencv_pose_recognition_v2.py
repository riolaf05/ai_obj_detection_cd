#Thanks to: https://blog.codecentric.de/en/2019/11/realtime-face-detection-and-filtering-with-the-coral-usb-accelerator/
import os
import io
import time
import base64
import logging
import socketserver
from http import server
from threading import Condition
import picamera

from imutils.video import VideoStream

import cv2
from PIL import Image
import glob

import numpy as np

from edgetpu.detection.engine import DetectionEngine
from pose_engine import PoseEngine

# Parameters
AUTH_USERNAME = os.environ.get('AUTH_USERNAME', os.getenv('USER'))
AUTH_PASSWORD = os.environ.get('AUTH_PASSWORD', os.getenv('PASSWORD'))
AUTH_BASE64 = base64.b64encode('{}:{}'.format(AUTH_USERNAME, AUTH_PASSWORD).encode('utf-8'))
BASIC_AUTH = 'Basic {}'.format(AUTH_BASE64.decode('utf-8'))
RESOLUTION = os.environ.get('RESOLUTION', '800x600').split('x')
RESOLUTION_X = int(RESOLUTION[0])
RESOLUTION_Y = int(RESOLUTION[1])
FRAMERATE = int(os.environ.get('FRAMERATE', '30'))
ROTATION = int(os.environ.get('ROTATE', 0))
HFLIP = os.environ.get('HFLIP', 'false').lower() == 'true'
VFLIP = os.environ.get('VFLIP', 'true').lower() == 'true'

PAGE = """\
<html>
<head>
<title>edgeTPU Object Detection</title>
</head>
<body>
<img src="stream.mjpg" width="{}" height="{}" />
</body>
</html>
""".format(RESOLUTION_X, RESOLUTION_Y)

engine = PoseEngine('/home/scripts/pose_detection/browser/models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.engine = None

    def set_engine(self, engine):
        self.engine = engine

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Authorization') is None:
            self.do_AUTHHEAD()
            self.wfile.write(b'no auth header received')
        elif self.headers.get('Authorization') == BASIC_AUTH:
            self.authorized_get()
        else:
            self.do_AUTHHEAD()
            self.wfile.write(b'not authenticated')

    def do_AUTHHEAD(self):
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm=\"picamera\"')
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def authorized_get(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                stream_video = io.BytesIO()
                frames_counter = 0

                
                while True:
                    # getting image
                    frames_counter += 1
    
                    camera.capture(stream_video, 
                                format='jpeg', 
                                use_video_port=True)
                    stream_video.truncate()
                    stream_video.seek(0)
                    
                    # cv2 / PIL coding
                    cv2_im = np.frombuffer(stream_video.getvalue(), dtype=np.uint8)
                    cv2_im = cv2.imdecode(cv2_im, 1)
                    pil_im = Image.fromarray(cv2_im)

                    pil_im.resize((641, 481), Image.NEAREST)
                    poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_im))

                    #print points on shell
                    for pose in poses:
                        if pose.score < 0.4: continue
                        print('\nPose Score: ', pose.score)
                        for label, keypoint in pose.keypoints.items():
                            print(' %-20s x=%-4d y=%-4d score=%.1f' %
                                (label, keypoint.yx[1], keypoint.yx[0], keypoint.score))

                    r, buf = cv2.imencode(".jpg", cv2_im)

                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-type','image/jpeg')
                    self.send_header('Content-length',str(len(buf)))
                    self.end_headers()
                    self.wfile.write(bytearray(buf))
                    self.wfile.write(b'\r\n')
                    

            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

if __name__ == '__main__':
    res = '{}x{}'.format(RESOLUTION_X, RESOLUTION_Y)

    with picamera.PiCamera(resolution=res, framerate=FRAMERATE, sensor_mode=2) as camera:
        camera.hflip = HFLIP
        camera.vflip = VFLIP
        camera.rotation = ROTATION

        try:
            address = ('', 8000)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        except:
            print("error on the server!")