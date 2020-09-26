import os
import numpy as np
from PIL import Image, ImageFont
import PIL.ImageDraw as ImageDraw,PIL.Image as Image, PIL.ImageShow as ImageShow
from pose_engine import PoseEngine
import cv2
import argparse

BASE_DIR='/home/scripts/pose_detection'

parser = argparse.ArgumentParser(description='Insert image file..')
parser.add_argument('--image', type=str, help='Image file path..', default='adam') 
args = parser.parse_args()

pil_image = Image.open(os.path.join(BASE_DIR, 'images', args.image))
pil_image.resize((641, 481), Image.NEAREST)
engine = PoseEngine(os.path.join(BASE_DIR,'models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'))
poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_image))
print('Inference time: %.fms' % inference_time)

cv2_img = cv2.imread(os.path.join(BASE_DIR, 'images', args.image))
cv2.resize(cv2_img, (641, 481))
for pose in poses:
    i=0
    if pose.score < 0.4: continue
    print('\nPose Score: ', pose.score)
    for label, keypoint in pose.keypoints.items():
        print(' %-20s x=%-4d y=%-4d score=%.1f' % (label, keypoint.yx[1], keypoint.yx[0], keypoint.score))

        #draw text
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (50, 50)
        # fontScale
        fontScale = 0.5
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        #cv2_img=cv2.putText(cv2_img, str(i), tuple([keypoint.yx[1], keypoint.yx[0]]), font, fontScale, color, thickness, cv2.LINE_AA)
        i+=1

        #draw points
        cv2.circle(cv2_img, tuple([keypoint.yx[1], keypoint.yx[0]]), 2, (0,0,255), -1)

cv2.imwrite(args.image[:-4]+"_out.jpg", cv2_img)